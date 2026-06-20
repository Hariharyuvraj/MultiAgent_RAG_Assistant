import logging
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import PrivateAttr

from config import load_config
from providers import get_llm
from schemas.state import AgentState

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    _llm: Any = PrivateAttr(default=None)
    _max_turns: int = PrivateAttr(default=10)
    _detect_followup: bool = PrivateAttr(default=True)

    def __init__(self) -> None:
        super().__init__(name="context_agent")
        cfg = load_config()
        self._llm = get_llm(cfg)
        self._max_turns = cfg["agents"]["context"]["max_history_turns"]
        self._detect_followup = cfg["agents"]["context"]["follow_up_detection"]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = AgentState.from_session_dict(dict(ctx.session.state))
        logger.info("[ContextAgent] session=%s query='%s'", state.session_id, state.query)

        if not state.history or not self._detect_followup:
            state.enriched_query = state.query
            logger.debug("[ContextAgent] No history — using raw query.")
        else:
            recent = state.history[-self._max_turns:]
            formatted = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}" for t in recent
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a conversation analyst. Given a conversation history "
                        "and a new user query, answer ONLY 'YES' if the new query is a "
                        "follow-up to the conversation, or 'NO' if it is a fresh topic."
                    )
                ),
                HumanMessage(
                    content=f"Conversation:\n{formatted}\n\nNew query: {state.query}"
                ),
            ]
            response = await self._llm.ainvoke(messages)
            is_followup = response.content.strip().upper().startswith("YES")
            logger.debug("[ContextAgent] follow-up detected: %s", is_followup)

            if is_followup:
                state.enriched_query = (
                    f"Context from prior conversation:\n{formatted}\n\n"
                    f"Current question: {state.query}"
                )
            else:
                state.enriched_query = state.query

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state.to_session_dict()),
        )
