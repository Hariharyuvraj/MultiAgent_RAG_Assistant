"""
Generate ground truth Q&A pairs from indexed documents using Groq LLM.
Saves to eval/ground_truth.csv with columns:
  question, ground_truth, source_document, category

Run:
    python eval/generate_gt.py
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_CSV = ROOT / "eval" / "ground_truth.csv"
QUESTIONS_PER_DOC = 30  # ~60 total across 2 docs

_SYSTEM_PROMPT = """\
You are a precise question-answer pair generator for RAG system evaluation.
Given a passage from a document, generate realistic user questions and their accurate answers.

Rules:
- Questions must be answerable ONLY from the given passage — no outside knowledge
- Answers must be factual, specific, and directly drawn from the passage
- Questions should be diverse: factual, procedural, numerical, definitional
- Do NOT generate vague or overly broad questions
- Return ONLY valid JSON — no markdown fences, no extra text

Output format (JSON array):
[
  {
    "question": "...",
    "ground_truth": "...",
    "category": "factual|procedural|numerical|definitional"
  }
]
"""

_USER_PROMPT = """\
Document: {doc_name}
Passage:
\"\"\"
{passage}
\"\"\"

Generate {n} question-answer pairs from this passage. Return JSON only.
"""


def _chunk_text(text: str, size: int = 1800, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _load_pdf_text(pdf_path: Path) -> str:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(str(pdf_path)).load()
        return "\n".join(d.page_content for d in docs)
    except Exception as e:
        print(f"  [warn] Could not load {pdf_path.name}: {e}")
        return ""


def _call_groq(client, passage: str, doc_name: str, n: int) -> list[dict]:
    prompt = _USER_PROMPT.format(doc_name=doc_name, passage=passage, n=n)
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"  [warn] Groq call failed: {e}")
        return []


def generate(docs_dir: Path, output_csv: Path, questions_per_doc: int = QUESTIONS_PER_DOC):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    pdf_files = list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt"))
    if not pdf_files:
        print(f"No PDF/TXT files found in {docs_dir}")
        return

    all_rows: list[dict] = []

    for pdf in pdf_files:
        print(f"\nProcessing: {pdf.name}")
        text = _load_pdf_text(pdf) if pdf.suffix == ".pdf" else pdf.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            print("  [skip] Empty text extracted.")
            continue

        chunks = _chunk_text(text, size=2000, overlap=200)
        # pick evenly spaced chunks to cover the whole document
        step = max(1, len(chunks) // questions_per_doc)
        selected = chunks[::step][:questions_per_doc]

        qs_per_chunk = max(1, questions_per_doc // len(selected))
        doc_rows: list[dict] = []

        for i, chunk in enumerate(selected):
            print(f"  chunk {i+1}/{len(selected)} → requesting {qs_per_chunk} Q&A pairs...")
            pairs = _call_groq(client, chunk, pdf.name, qs_per_chunk)
            for pair in pairs:
                if "question" in pair and "ground_truth" in pair:
                    doc_rows.append({
                        "question":       pair["question"].strip(),
                        "ground_truth":   pair["ground_truth"].strip(),
                        "source_document": pdf.name,
                        "category":       pair.get("category", "factual"),
                    })
            time.sleep(0.5)  # stay within rate limits

        print(f"  Generated {len(doc_rows)} pairs from {pdf.name}")
        all_rows.extend(doc_rows)

    if not all_rows:
        print("No Q&A pairs generated.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "source_document", "category"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} Q&A pairs → {output_csv}")
    _print_sample(all_rows[:3])


def _print_sample(rows: list[dict]):
    print("\n--- Sample Q&A pairs ---")
    for r in rows:
        print(f"\n[{r['category'].upper()}] {r['source_document']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['ground_truth'][:120]}...")


if __name__ == "__main__":
    docs_dir = ROOT / "documents"
    generate(docs_dir, OUTPUT_CSV)
