import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from langchain_community.vectorstores import Chroma
from pydantic import PrivateAttr

from config import load_config
from ingest.embedder import Embedder
from providers import get_embeddings
from schemas.state import AgentState, ChunkResult

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    _embedder: Any = PrivateAttr(default=None)
    _top_k: int = PrivateAttr(default=5)
    _threshold: float = PrivateAttr(default=0.35)
    _vectorstore: Optional[Any] = PrivateAttr(default=None)

    def __init__(self) -> None:
        super().__init__(name="search_agent")
        cfg = load_config()
        embeddings = get_embeddings(cfg)
        vs_cfg = cfg["vector_store"]
        self._embedder = Embedder(
            embeddings=embeddings,
            persist_path=vs_cfg["persist_path"],
            collection_name=vs_cfg["collection_name"],
        )
        ret_cfg = cfg["retrieval"]
        self._top_k = ret_cfg["top_k"]
        self._threshold = ret_cfg["score_threshold"]

    def _get_store(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = self._embedder.load_store()
        return self._vectorstore

    def _deduplicate(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        seen: Dict[Tuple[str, int], ChunkResult] = {}
        for chunk in chunks:
            key = (chunk.source, chunk.page)
            if key not in seen or chunk.score > seen[key].score:
                seen[key] = chunk
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = AgentState.from_session_dict(dict(ctx.session.state))
        logger.info("[SearchAgent] Executing %d sub-queries.", len(state.plan))

        store = self._get_store()
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self._top_k,
                "score_threshold": self._threshold,
            },
        )

        all_chunks: List[ChunkResult] = []
        for sub_query in state.plan:
            docs = retriever.invoke(sub_query)
            for doc in docs:
                chunk = ChunkResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=int(doc.metadata.get("page", 0)),
                    score=float(doc.metadata.get("score", 0.0)),
                )
                all_chunks.append(chunk)
            logger.debug("[SearchAgent] sub-query='%s' hits=%d", sub_query, len(docs))

        state.retrieved_chunks = self._deduplicate(all_chunks)
        logger.info("[SearchAgent] Retrieved %d unique chunks.", len(state.retrieved_chunks))

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state.to_session_dict()),
        )
