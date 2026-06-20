import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.state import AgentState, ChunkResult


def _base_state(**kwargs) -> AgentState:
    defaults = dict(
        query="What is RAG?",
        session_id="test-session",
        history=[],
        enriched_query="What is RAG?",
        plan=[],
        retrieved_chunks=[],
        answer="",
        sources=[],
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


def _mock_ctx(state: AgentState):
    ctx = MagicMock()
    ctx.session.state = state.to_session_dict()
    return ctx


@pytest.mark.asyncio
async def test_context_agent_empty_history():
    from agents.context_agent import ContextAgent

    with patch("agents.context_agent.get_llm"), \
         patch("agents.context_agent.load_config", return_value={
             "llm": {"provider": "groq", "model_name": "x", "temperature": 0.2, "max_tokens": 512, "timeout": 30},
             "agents": {"context": {"max_history_turns": 10, "follow_up_detection": True}},
         }):
        agent = ContextAgent.__new__(ContextAgent)
        agent._llm = MagicMock()
        agent._max_turns = 10
        agent._detect_followup = True
        agent.name = "context_agent"

        state = _base_state()
        ctx = _mock_ctx(state)
        events = [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert result.enriched_query == "What is RAG?"
    assert len(events) == 1


@pytest.mark.asyncio
async def test_context_agent_enriches_with_history():
    from agents.context_agent import ContextAgent
    from langchain_core.messages import AIMessage

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="YES")

    with patch("agents.context_agent.get_llm"), \
         patch("agents.context_agent.load_config", return_value={
             "llm": {"provider": "groq", "model_name": "x", "temperature": 0.2, "max_tokens": 512, "timeout": 30},
             "agents": {"context": {"max_history_turns": 10, "follow_up_detection": True}},
         }):
        agent = ContextAgent.__new__(ContextAgent)
        agent._llm = mock_llm
        agent._max_turns = 10
        agent._detect_followup = True
        agent.name = "context_agent"

        state = _base_state(history=[{"user": "Hello", "assistant": "Hi there"}])
        ctx = _mock_ctx(state)
        [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert "Hello" in result.enriched_query or "Context" in result.enriched_query


@pytest.mark.asyncio
async def test_planner_agent_returns_list():
    from agents.planner_agent import PlannerAgent
    from langchain_core.messages import AIMessage

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(
        content="1. Search for RAG definition\n2. Find retrieval details"
    )

    with patch("agents.planner_agent.get_llm"), \
         patch("agents.planner_agent.load_config", return_value={
             "llm": {"provider": "groq", "model_name": "x", "temperature": 0.2, "max_tokens": 512, "timeout": 30},
             "agents": {"planner": {"max_steps": 4, "output_format": "numbered_list"}},
         }):
        agent = PlannerAgent.__new__(PlannerAgent)
        agent._llm = mock_llm
        agent._max_steps = 4
        agent.name = "planner_agent"

        state = _base_state(enriched_query="What is RAG?")
        ctx = _mock_ctx(state)
        [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert isinstance(result.plan, list)
    assert len(result.plan) >= 1


@pytest.mark.asyncio
async def test_analyst_agent_no_chunks():
    from agents.analyst_agent import AnalystAgent

    with patch("agents.analyst_agent.get_llm"), \
         patch("agents.analyst_agent.load_config", return_value={
             "llm": {"provider": "groq", "model_name": "x", "temperature": 0.2, "max_tokens": 512, "timeout": 30},
         }):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent._llm = MagicMock()
        agent.name = "analyst_agent"

        state = _base_state(retrieved_chunks=[])
        ctx = _mock_ctx(state)
        [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert result.answer != ""


@pytest.mark.asyncio
async def test_guard_agent_passes_high_grounding():
    from agents.guard_agent import GuardAgent

    with patch("agents.guard_agent.load_config", return_value={
        "agents": {"guard": {"grounding_threshold": 0.60, "max_retries": 2, "fallback_message": "fallback"}},
    }), patch("agents.guard_agent.compute_grounding_score", return_value=0.85):
        agent = GuardAgent.__new__(GuardAgent)
        agent._threshold = 0.60
        agent._max_retries = 2
        agent._fallback = "fallback"
        agent.name = "guard_agent"

        state = _base_state(answer="test answer", retrieved_chunks=[
            ChunkResult(content="chunk", source="test.pdf", page=1, score=0.8)
        ])
        ctx = _mock_ctx(state)
        [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert result.passed_guard is True
    assert result.grounding_score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_guard_agent_retries_on_low_grounding():
    from agents.guard_agent import GuardAgent

    with patch("agents.guard_agent.load_config", return_value={
        "agents": {"guard": {"grounding_threshold": 0.60, "max_retries": 2, "fallback_message": "fallback"}},
    }), patch("agents.guard_agent.compute_grounding_score", return_value=0.30):
        agent = GuardAgent.__new__(GuardAgent)
        agent._threshold = 0.60
        agent._max_retries = 2
        agent._fallback = "fallback"
        agent.name = "guard_agent"

        state = _base_state(answer="test answer", retry_count=0, retrieved_chunks=[
            ChunkResult(content="chunk", source="test.pdf", page=1, score=0.3)
        ])
        ctx = _mock_ctx(state)
        [e async for e in agent._run_async_impl(ctx)]

    result = AgentState.from_session_dict(ctx.session.state)
    assert result.passed_guard is False
    assert result.retry_count == 1
