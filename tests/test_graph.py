from unittest.mock import patch, MagicMock

import pytest


def test_build_pipeline_returns_sequential_agent():
    from google.adk.agents import SequentialAgent

    with patch("graph.agent_graph.ContextAgent"), \
         patch("graph.agent_graph.PlannerAgent"), \
         patch("graph.agent_graph.SearchAgent"), \
         patch("graph.agent_graph.AnalystAgent"), \
         patch("graph.agent_graph.GuardAgent"):

        from graph.agent_graph import build_pipeline
        pipeline = build_pipeline()

    assert isinstance(pipeline, SequentialAgent)
    assert len(pipeline.sub_agents) == 5
