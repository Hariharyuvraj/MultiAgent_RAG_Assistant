import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(name="MultiAgent_RAG_Assistant", add_completion=False)
console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )


@app.command("ingest")
def ingest():
    """Index all documents in documents/ into ChromaDB."""
    _setup_logging()
    from config import load_config
    from ingest.doc_loader import DocumentLoader
    from ingest.text_splitter import DocumentSplitter
    from ingest.embedder import Embedder
    from providers import get_embeddings

    cfg = load_config()
    ingest_cfg = cfg["ingest"]
    vs_cfg = cfg["vector_store"]

    console.print("[bold]Loading documents...[/bold]")
    loader = DocumentLoader(ingest_cfg["docs_path"], ingest_cfg["supported_formats"])
    docs = loader.load()

    if not docs:
        console.print("[yellow]No documents found. Drop PDF/TXT files into documents/[/yellow]")
        raise typer.Exit(code=1)

    splitter = DocumentSplitter(ingest_cfg["chunk_size"], ingest_cfg["chunk_overlap"])
    chunks = splitter.split(docs)

    embedder = Embedder(
        embeddings=get_embeddings(cfg),
        persist_path=vs_cfg["persist_path"],
        collection_name=vs_cfg["collection_name"],
    )
    embedder.embed_and_store(chunks)

    source_names = {d.metadata.get("source", "unknown") for d in docs}
    console.print(
        f"[green]Indexed {len(chunks)} chunks from {len(source_names)} document(s).[/green]"
    )


@app.command("query")
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override LLM provider (groq|gemini)."),
):
    """Run a single query through the full pipeline."""
    _setup_logging()
    import os
    if provider:
        os.environ["LLM_PROVIDER"] = provider

    from config import load_config
    from eval.trace_logger import TraceLogger
    from graph.agent_graph import build_pipeline
    from schemas.state import AgentState

    cfg = load_config()
    pipeline = build_pipeline()
    trace_logger = TraceLogger(cfg["eval"]["trace_dir"])

    session_id = str(uuid.uuid4())[:8]
    state = AgentState(
        query=question,
        session_id=session_id,
        history=[],
    )

    async def _run():
        from app import run_query
        return await run_query(pipeline, state)

    t_start = time.monotonic()
    result = asyncio.run(_run())
    latency_ms = int((time.monotonic() - t_start) * 1000)

    console.print(f"\n[bold cyan]Answer:[/bold cyan]\n{result.answer}")
    if result.sources:
        console.print(f"\n[dim]Sources: {', '.join(result.sources)}[/dim]")
    console.print(
        f"[dim]Grounding: {result.grounding_score:.2f} | Latency: {latency_ms} ms[/dim]"
    )

    if cfg["eval"]["enabled"]:
        trace_logger.write(result, latency_ms)


@app.command("eval")
def eval_report():
    """Print evaluation metrics from saved traces."""
    _setup_logging()
    from config import load_config
    from eval.metrics import MetricsReporter

    cfg = load_config()
    reporter = MetricsReporter(cfg["eval"]["trace_dir"])
    reporter.print_summary()


@app.command("generate-gt")
def generate_ground_truth(
    output: str = typer.Option("eval/ground_truth.csv", "--output", help="Output CSV path"),
    per_doc: int = typer.Option(30, "--per-doc", help="Questions to generate per document"),
):
    """Generate ground truth Q&A pairs from documents/ using Groq LLM."""
    _setup_logging()
    from pathlib import Path
    from eval.generate_gt import generate
    generate(Path("documents"), Path(output), questions_per_doc=per_doc)


@app.command("ragas")
def ragas_eval(
    gt_csv: str = typer.Option("eval/ground_truth.csv", "--gt", help="Ground truth CSV path"),
    output: str = typer.Option("eval/ragas_results.json", "--output", help="Results JSON path"),
    max_q: int = typer.Option(60, "--max-q", help="Max questions to evaluate"),
):
    """Run RAGAS evaluation against ground truth dataset."""
    _setup_logging()
    from pathlib import Path
    from eval.ragas_runner import run_ragas_eval
    run_ragas_eval(gt_csv=Path(gt_csv), results_json=Path(output), max_questions=max_q)


if __name__ == "__main__":
    app()
