import json
import logging
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


def _load_traces(traces_dir: str) -> List[dict]:
    path = Path(traces_dir)
    traces = []
    for f in sorted(path.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                traces.append(json.load(fh))
        except Exception:
            logger.warning("Could not read trace file: %s", f)
    return traces


def compute_avg_retrieval_score(trace: dict) -> float:
    chunks = trace.get("retrieved_chunks", [])
    if not chunks:
        return 0.0
    return sum(c.get("score", 0.0) for c in chunks) / len(chunks)


def compute_grounding_rate(traces_dir: str) -> float:
    traces = _load_traces(traces_dir)
    if not traces:
        return 0.0
    passed = sum(1 for t in traces if t.get("passed_guard", False))
    return passed / len(traces) * 100.0


def compute_avg_latency(traces_dir: str) -> float:
    traces = _load_traces(traces_dir)
    latencies = [t.get("latency_ms", 0) for t in traces if "latency_ms" in t]
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


class MetricsReporter:
    def __init__(self, traces_dir: str) -> None:
        self.traces_dir = traces_dir

    def print_summary(self) -> None:
        traces = _load_traces(self.traces_dir)
        if not traces:
            console.print("[yellow]No traces found in:[/yellow]", self.traces_dir)
            return

        grounding_rate = compute_grounding_rate(self.traces_dir)
        avg_latency = compute_avg_latency(self.traces_dir)
        avg_retrieval = sum(compute_avg_retrieval_score(t) for t in traces) / len(traces)

        table = Table(title="MultiAgent_RAG_Assistant — Evaluation Summary", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Total Queries", str(len(traces)))
        table.add_row("Grounding Pass Rate", f"{grounding_rate:.1f}%")
        table.add_row("Avg Retrieval Score", f"{avg_retrieval:.3f}")
        table.add_row("Avg Latency", f"{avg_latency:.0f} ms")
        console.print(table)
