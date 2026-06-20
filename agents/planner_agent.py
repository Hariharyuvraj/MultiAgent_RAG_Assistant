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
    "You are a query planning expert. Given a user question about documents, "
    "break it into 2 to 4 specific retrieval sub-steps. Each step must be a focused "
    "search query. Return ONLY a numbered list, one item per line. "
    "Example:\n1. Search for the definition of X\n2. Find details about Y"
)


def _parse_plan(text: str) -> List[str]:
    lines = text.strip().splitlines()
    plan = []
    for line in lines:
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            plan.append(cleaned)
    return plan or [text.strip()]


class PlannerAgent(BaseAgent):
    _llm: Any = PrivateAttr(default=None)
    _max_steps: int = PrivateAttr(default=4)

    def __init__(self) -> None:
        super().__init__(name="planner_agent")
        cfg = load_config()
        self._llm = get_llm(cfg)
        self._max_steps = cfg["agents"]["planner"]["max_steps"]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = AgentState.from_session_dict(dict(ctx.session.state))
        logger.info("[PlannerAgent] Planning for query: '%s'", state.enriched_query)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=state.enriched_query),
        ]
        response = await self._llm.ainvoke(messages)
        plan = _parse_plan(response.content)[: self._max_steps]
        state.plan = plan
        logger.info("[PlannerAgent] Plan: %s", plan)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state.to_session_dict()),
        )
