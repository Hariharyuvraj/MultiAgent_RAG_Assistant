import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import PrivateAttr

from config import load_config
from guardrails.grounding_check import compute_grounding_score
from schemas.state import AgentState

logger = logging.getLogger(__name__)


class GuardAgent(BaseAgent):
    _threshold: float = PrivateAttr(default=0.60)
    _max_retries: int = PrivateAttr(default=2)
    _fallback: str = PrivateAttr(default="")

    def __init__(self) -> None:
        super().__init__(name="guard_agent")
        cfg = load_config()
        guard_cfg = cfg["agents"]["guard"]
        self._threshold = float(guard_cfg["grounding_threshold"])
        self._max_retries = int(guard_cfg["max_retries"])
        self._fallback = guard_cfg["fallback_message"]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = AgentState.from_session_dict(dict(ctx.session.state))
        logger.info("[GuardAgent] Scoring answer grounding.")

        score = compute_grounding_score(state.answer, state.retrieved_chunks)
        state.grounding_score = score
        logger.info(
            "[GuardAgent] grounding_score=%.3f threshold=%.3f retries=%d",
            score,
            self._threshold,
            state.retry_count,
        )

        if score >= self._threshold:
            state.passed_guard = True
            logger.info("[GuardAgent] PASS — answer accepted.")
        elif state.retry_count < self._max_retries:
            state.retry_count += 1
            state.passed_guard = False
            logger.warning(
                "[GuardAgent] RETRY %d/%d — score below threshold.",
                state.retry_count,
                self._max_retries,
            )
        else:
            state.passed_guard = False
            state.answer = self._fallback
            state.sources = []
            logger.warning("[GuardAgent] FALLBACK — max retries exhausted.")

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state.to_session_dict()),
        )
