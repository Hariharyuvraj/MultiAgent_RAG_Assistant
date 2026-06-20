import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from config import load_config
from eval.trace_logger import TraceLogger
from graph.agent_graph import build_pipeline
from schemas.state import AgentState

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


async def run_query(pipeline, state: AgentState):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="MultiAgent_RAG_Assistant",
        user_id=state.session_id,
        session_id=state.session_id,
        state=state.to_session_dict(),
    )

    runner = Runner(
        agent=pipeline,
        app_name="MultiAgent_RAG_Assistant",
        session_service=session_service,
    )

    user_content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=state.query)]
    )

    accumulated = state.to_session_dict()
    async for event in runner.run_async(
        user_id=state.session_id,
        session_id=state.session_id,
        new_message=user_content,
    ):
        if event.actions and event.actions.state_delta:
            accumulated.update(event.actions.state_delta)

    return AgentState.from_session_dict(accumulated)


async def main():
    cfg = load_config()
    pipeline = build_pipeline()
    trace_logger = TraceLogger(cfg["eval"]["trace_dir"])
    session_id = str(uuid.uuid4())[:8]
    history = []

    console.print("\n[bold]MultiAgent RAG Assistant[/bold]")
    console.print("Type your question or [bold]exit[/bold] to quit.\n")

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            console.print("Goodbye.")
            break

        state = AgentState(
            query=query,
            session_id=session_id,
            history=history,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        t_start = time.monotonic()
        try:
            result = await run_query(pipeline, state)
        except Exception:
            logger.exception("Pipeline error.")
            console.print("[red]An error occurred. Check logs.[/red]")
            continue
        latency_ms = int((time.monotonic() - t_start) * 1000)

        console.print("\n[bold cyan]Assistant:[/bold cyan]")
        console.print(Markdown(result.answer))
        if result.sources:
            console.print(f"\n[dim]Sources: {', '.join(result.sources)}[/dim]")
        console.print(
            f"[dim]Grounding: {result.grounding_score:.2f} | "
            f"Latency: {latency_ms} ms[/dim]\n"
        )

        if cfg["eval"]["enabled"]:
            trace_logger.write(result, latency_ms)

        history.append({"user": query, "assistant": result.answer})


if __name__ == "__main__":
    asyncio.run(main())
