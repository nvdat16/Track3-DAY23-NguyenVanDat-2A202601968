"""Reusable execution helpers for the CLI and Streamlit demo."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .metrics import MetricsReport, metric_from_state, summarize_metrics
from .state import AgentState, ApprovalDecision, Route, Scenario, initial_state

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

ProgressCallback = Callable[[int, int, Scenario], None]


def invoke_ticket(
    graph: CompiledStateGraph,
    query: str,
    max_attempts: int = 3,
    interrupt_enabled: bool = True,
) -> tuple[dict[str, Any], RunnableConfig]:
    """Start a single support workflow with a unique checkpoint thread."""
    scenario = Scenario(
        id=f"demo-{uuid4().hex[:10]}",
        query=query,
        expected_route=Route.SIMPLE,
        max_attempts=max_attempts,
    )
    state = initial_state(scenario)
    state["interrupt_enabled"] = interrupt_enabled
    config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
    result = cast(dict[str, Any], graph.invoke(state, config=config))
    return result, config


def resume_ticket(
    graph: CompiledStateGraph,
    config: RunnableConfig,
    approved: bool,
    comment: str = "",
) -> dict[str, Any]:
    """Resume a paused approval node with a human decision."""
    decision = ApprovalDecision(
        approved=approved,
        reviewer="streamlit-reviewer",
        comment=comment,
    )
    command: Command[object] = Command(resume=decision.model_dump())
    return cast(dict[str, Any], graph.invoke(command, config=config))


def interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first LangGraph interrupt payload from an invocation result."""
    interrupts = state.get("__interrupt__") or []
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else {"value": value}


def run_scenario_suite(
    graph: CompiledStateGraph,
    scenarios: list[Scenario],
    persistent_backend: bool,
    progress: ProgressCallback | None = None,
) -> tuple[MetricsReport, list[RunnableConfig]]:
    """Run grading scenarios with deterministic mock approval and collect metrics."""
    metrics = []
    run_configs: list[RunnableConfig] = []
    checkpoint_reads = []
    total = len(scenarios)
    for index, scenario in enumerate(scenarios, start=1):
        state: AgentState = initial_state(scenario)
        state["thread_id"] = f"{state['thread_id']}-{uuid4().hex[:8]}"
        state["interrupt_enabled"] = False
        run_config: RunnableConfig = {
            "configurable": {"thread_id": state["thread_id"]},
        }
        started_at = perf_counter()
        final_state = cast(dict[str, Any], graph.invoke(state, config=run_config))
        item = metric_from_state(
            final_state,
            scenario.expected_route.value,
            scenario.requires_approval,
        )
        item.latency_ms = round((perf_counter() - started_at) * 1000)
        metrics.append(item)
        run_configs.append(run_config)
        checkpoint_reads.append(bool(graph.get_state(run_config).values))
        if progress is not None:
            progress(index, total, scenario)

    resume_success = persistent_backend and bool(checkpoint_reads) and all(checkpoint_reads)
    return summarize_metrics(metrics, resume_success=resume_success), run_configs


def verify_checkpoint_recovery(
    kind: str,
    database_url: str | None,
    run_configs: list[RunnableConfig],
) -> bool:
    """Rebuild the graph and verify saved threads through a new checkpointer instance."""
    if kind not in {"sqlite", "postgres"} or not run_configs:
        return False

    from .graph import build_graph
    from .persistence import build_checkpointer

    checkpointer = build_checkpointer(kind, database_url)
    recovered_graph = build_graph(checkpointer)
    try:
        return all(bool(recovered_graph.get_state(config).values) for config in run_configs)
    finally:
        connection = getattr(checkpointer, "conn", None)
        if connection is not None:
            connection.close()
