import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from schemas.state import AgentState

logger = logging.getLogger(__name__)


class TraceLogger:
    def __init__(self, trace_dir: str) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write(self, state: AgentState, latency_ms: int) -> Path:
        timestamp = datetime.now(timezone.utc).isoformat()
        trace = {
            "session_id": state.session_id,
            "timestamp": timestamp,
            "query": state.query,
            "enriched_query": state.enriched_query,
            "plan": state.plan,
            "retrieved_chunks": [
                {
                    "source": c.source,
                    "page": c.page,
                    "score": c.score,
                    "content": c.content,
                }
                for c in state.retrieved_chunks
            ],
            "answer": state.answer,
            "sources": state.sources,
            "grounding_score": state.grounding_score,
            "passed_guard": state.passed_guard,
            "retry_count": state.retry_count,
            "latency_ms": latency_ms,
        }
        filename = f"{state.session_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        file_path = self.trace_dir / filename
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(trace, fh, indent=2, ensure_ascii=False)
        logger.debug("Trace written to %s", file_path)
        return file_path
