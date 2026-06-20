import logging

from google.adk.agents import SequentialAgent

from agents.context_agent import ContextAgent
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.analyst_agent import AnalystAgent
from agents.guard_agent import GuardAgent

logger = logging.getLogger(__name__)


def build_pipeline() -> SequentialAgent:
    logger.info("Building MultiAgent_RAG_Assistant pipeline.")
    pipeline = SequentialAgent(
        name="rag_pipeline",
        sub_agents=[
            ContextAgent(),
            PlannerAgent(),
            SearchAgent(),
            AnalystAgent(),
            GuardAgent(),
        ],
    )
    logger.info("Pipeline ready with %d agents.", len(pipeline.sub_agents))
    return pipeline
