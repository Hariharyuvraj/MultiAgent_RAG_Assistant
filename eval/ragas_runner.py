"""
RAGAS evaluation runner.

Reads eval/ground_truth.csv, runs each question through the RAG pipeline,
collects (question, answer, contexts, ground_truth), then computes RAGAS metrics.

Run via CLI:
    python cli.py ragas

Or directly:
    python eval/ragas_runner.py
"""
import asyncio
import csv
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GROUND_TRUTH_CSV = ROOT / "eval" / "ground_truth.csv"
RESULTS_JSON     = ROOT / "eval" / "ragas_results.json"

console = Console()


# ─────────────────────────────────────────────
# 1. Load ground truth
# ─────────────────────────────────────────────

def load_ground_truth(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        console.print(f"[red]Ground truth CSV not found: {csv_path}[/red]")
        console.print("Run:  python eval/generate_gt.py")
        sys.exit(1)
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("question") and row.get("ground_truth"):
                rows.append(row)
    return rows


# ─────────────────────────────────────────────
# 2. Run pipeline on each question
# ─────────────────────────────────────────────

async def _run_single(pipeline, question: str, session_id: str) -> dict:
    from schemas.state import AgentState
    state = AgentState(query=question, session_id=session_id, history=[])

    # collect streamed tokens
    answer_tokens: list[str] = []

    async for event in pipeline.run_async(
        input={"state": state.model_dump()},
        session_id=session_id,
    ):
        output = getattr(event, "output", None) or {}
        if isinstance(output, dict):
            s = output.get("state", {})
            if s.get("answer"):
                answer_tokens = [s["answer"]]
            state_data = s

    final_state = state.model_dump()
    final_state.update(state_data if "state_data" in dir() else {})
    return final_state


def run_pipeline_on_questions(gt_rows: list[dict], max_questions: int = 60) -> list[dict]:
    from pipeline.rag_pipeline import RAGPipeline

    total = min(len(gt_rows), max_questions)
    console.print(f"\n[bold]Running pipeline on {total} questions (this takes ~{total*7//60} min)...[/bold]")
    rag = RAGPipeline()

    results = []
    for i, row in enumerate(gt_rows[:max_questions]):
        question = row["question"]
        ground_truth = row["ground_truth"]

        console.print(f"  [{i+1}/{total}] {question[:75]}...")

        t_start = time.monotonic()
        try:
            state = rag.run_sync(question)
        except Exception as e:
            console.print(f"    [red]Pipeline error: {e}[/red]")
            state = None
        latency_ms = int((time.monotonic() - t_start) * 1000)

        if state is None:
            results.append({
                "question":        question,
                "answer":          "",
                "contexts":        [],
                "ground_truth":    ground_truth,
                "source_document": row.get("source_document", ""),
                "category":        row.get("category", "factual"),
                "latency_ms":      latency_ms,
                "grounding_score": 0.0,
                "is_web_answer":   False,
            })
            continue

        contexts = [c.content for c in state.retrieved_chunks] if state.retrieved_chunks else []

        results.append({
            "question":        question,
            "answer":          state.answer,
            "contexts":        contexts,
            "ground_truth":    ground_truth,
            "source_document": row.get("source_document", ""),
            "category":        row.get("category", "factual"),
            "latency_ms":      latency_ms,
            "grounding_score": state.grounding_score,
            "is_web_answer":   state.is_web_answer,
        })

    return results


# ─────────────────────────────────────────────
# 3. Compute RAGAS metrics
# ─────────────────────────────────────────────

def compute_ragas(results: list[dict]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        console.print("[red]ragas not installed. Run:  pip install ragas datasets[/red]")
        sys.exit(1)

    # RAGAS skips rows with empty answer or empty contexts
    valid = [r for r in results if r["answer"] and r["contexts"]]
    if not valid:
        console.print("[yellow]No valid results to evaluate (all answers or contexts empty).[/yellow]")
        return {}

    console.print(f"\n[bold]Running RAGAS on {len(valid)} valid samples...[/bold]")
    console.print("[dim]Using Groq (llama-3.1-8b-instant) + HuggingFace embeddings — no OpenAI needed[/dim]\n")

    # ── Wire RAGAS to use Groq instead of OpenAI ──────────────────────────────
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    ragas_llm  = LangchainLLMWrapper(groq_llm)
    ragas_emb  = LangchainEmbeddingsWrapper(hf_embeddings)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in valid
    ])

    score = evaluate(dataset, metrics=metrics)
    return dict(score)


# ─────────────────────────────────────────────
# 4. Display + save results
# ─────────────────────────────────────────────

def print_ragas_scores(scores: dict, results: list[dict]):
    table = Table(title="RAGAS Evaluation Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Interpretation")

    _interp = {
        "faithfulness":        ("< 0.6 hallucination risk", "0.6–0.8 acceptable", "> 0.8 excellent"),
        "answer_relevancy":    ("< 0.6 off-topic answers",  "0.6–0.8 acceptable", "> 0.8 excellent"),
        "context_precision":   ("< 0.5 noisy retrieval",    "0.5–0.75 acceptable","> 0.75 excellent"),
        "context_recall":      ("< 0.5 missing info",       "0.5–0.75 acceptable","> 0.75 excellent"),
        "answer_correctness":  ("< 0.5 wrong answers",      "0.5–0.75 acceptable", "> 0.75 excellent"),
    }

    for metric, value in scores.items():
        if not isinstance(value, float):
            continue
        low, mid, high = _interp.get(metric, ("", "", ""))
        interp = high if value > 0.75 else (mid if value > 0.5 else low)
        table.add_row(metric.replace("_", " ").title(), f"{value:.4f}", interp)

    console.print(table)

    # summary stats
    total   = len(results)
    web_ans = sum(1 for r in results if r.get("is_web_answer"))
    valid   = sum(1 for r in results if r["answer"] and r["contexts"])
    avg_lat = sum(r["latency_ms"] for r in results) / total if total else 0

    console.print(f"\n[dim]Total questions: {total} | RAGAS-evaluated: {valid} | "
                  f"Web fallback: {web_ans} | Avg latency: {avg_lat:.0f} ms[/dim]")


def save_results(scores: dict, results: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ragas_scores": scores,
        "per_question":  results,
        "summary": {
            "total":        len(results),
            "evaluated":    sum(1 for r in results if r["answer"] and r["contexts"]),
            "web_fallback": sum(1 for r in results if r.get("is_web_answer")),
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]Results saved → {output_path}[/green]")


# ─────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────

def run_ragas_eval(
    gt_csv: Path = GROUND_TRUTH_CSV,
    results_json: Path = RESULTS_JSON,
    max_questions: int = 60,
):
    gt_rows = load_ground_truth(gt_csv)
    console.print(f"[bold]Loaded {len(gt_rows)} ground truth pairs[/bold]")

    results = run_pipeline_on_questions(gt_rows, max_questions=max_questions)
    scores  = compute_ragas(results)

    if scores:
        print_ragas_scores(scores, results)

    save_results(scores, results, results_json)
    return scores


if __name__ == "__main__":
    run_ragas_eval()
