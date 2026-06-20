import logging
import re
from typing import Any, AsyncGenerator, List

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import PrivateAttr

from config import load_config
from providers import get_llm
from schemas.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document analyst. Answer the user's question using ONLY the provided "
    "document excerpts. Always cite the source document name and page number inline "
    "using the format [source:page]. If the answer cannot be determined from the "
    "provided excerpts, state that clearly — do not guess or use outside knowledge."
)


def _build_context(state: AgentState) -> str:
    parts = []
    for idx, chunk in enumerate(state.retrieved_chunks, start=1):
        parts.append(
            f"[{idx}] Source: {chunk.source}, Page: {chunk.page}\n{chunk.content}"
        )
    return "\n\n".join(parts)


def _extract_sources(answer: str, state: AgentState) -> List[str]:
    cited = set()
    for chunk in state.retrieved_chunks:
        pattern = re.escape(chunk.source)
        if re.search(pattern, answer, re.IGNORECASE):
            cited.add(f"{chunk.source}:{chunk.page}")
    return sorted(cited) if cited else [
        f"{c.source}:{c.page}" for c in state.retrieved_chunks
    ]


class AnalystAgent(BaseAgent):
    _llm: Any = PrivateAttr(default=None)

    def __init__(self) -> None:
        super().__init__(name="analyst_agent")
        cfg = load_config()
        self._llm = get_llm(cfg)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = AgentState.from_session_dict(dict(ctx.session.state))
        logger.info(
            "[AnalystAgent] Generating answer from %d chunks.",
            len(state.retrieved_chunks),
        )

        if not state.retrieved_chunks:
            state.answer = "No relevant document content was retrieved for this query."
            state.sources = []
            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state.to_session_dict()),
            )
            return

        context_block = _build_context(state)
        user_message = (
            f"Question: {state.enriched_query}\n\nDocument Excerpts:\n{context_block}"
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        state.answer = response.content.strip()
        state.sources = _extract_sources(state.answer, state)
        logger.info("[AnalystAgent] Answer generated. Sources: %s", state.sources)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state.to_session_dict()),
        )
