"""Streamlit operations console for the LangGraph agent lab."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import MetricsReport, write_metrics
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.report import render_report, write_report
from langgraph_agent_lab.runner import (
    interrupt_payload,
    invoke_ticket,
    resume_ticket,
    run_scenario_suite,
)
from langgraph_agent_lab.scenarios import load_scenarios

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
METRICS_PATH = DEMO_ROOT / "outputs" / "metrics.json"
REPORT_PATH = DEMO_ROOT / "outputs" / "lab_report.md"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "sample" / "scenarios.jsonl"

SAMPLE_QUERIES = {
    "Password help": "How do I reset my password?",
    "Order lookup": "Please lookup order status for order 12345",
    "Missing context": "Can you fix it?",
    "Refund approval": "Refund this customer and send confirmation email",
    "Transient failure": "Timeout failure while processing request",
    "Delete account": "Delete customer account after support verification",
    "Custom": "",
}

GRAPH_DOT = """
digraph LangGraphLab {
  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.55"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10,
        color="#cbd5e1", penwidth=1.2, margin="0.12,0.08"];
  edge [color="#64748b", arrowsize=0.65, fontname="Arial", fontsize=9];

  start [label="START", shape=circle, fillcolor="#f1f5f9"];
  intake [fillcolor="#e2e8f0"];
  classify [fillcolor="#fef3c7", color="#d97706"];
  answer [fillcolor="#dcfce7", color="#16a34a"];
  tool [fillcolor="#ccfbf1", color="#0f766e"];
  evaluate [fillcolor="#e0f2fe", color="#0284c7"];
  clarify [fillcolor="#f1f5f9"];
  risky_action [label="risky action", fillcolor="#ffedd5", color="#ea580c"];
  approval [fillcolor="#ffedd5", color="#ea580c"];
  retry [fillcolor="#fee2e2", color="#dc2626"];
  dead_letter [label="dead letter", fillcolor="#fee2e2", color="#991b1b"];
  finalize [fillcolor="#dcfce7", color="#15803d"];
  end [label="END", shape=circle, fillcolor="#f1f5f9"];

  start -> intake -> classify;
  classify -> answer [label="simple"];
  classify -> tool [label="tool"];
  classify -> clarify [label="missing"];
  classify -> risky_action [label="risky"];
  classify -> retry [label="error"];
  risky_action -> approval;
  approval -> tool [label="approve"];
  approval -> clarify [label="reject"];
  tool -> evaluate;
  evaluate -> answer [label="success"];
  evaluate -> retry [label="retry"];
  retry -> tool [label="within limit"];
  retry -> dead_letter [label="exhausted"];
  answer -> finalize;
  clarify -> finalize;
  dead_letter -> finalize;
  finalize -> end;
}
"""


def _configure_page() -> None:
    st.set_page_config(
        page_title="LangGraph Support Console",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1280px; padding-top: 1.4rem; padding-bottom: 2rem;}
        h1 {font-size: 1.75rem !important; letter-spacing: 0 !important;}
        h2, h3 {letter-spacing: 0 !important;}
        div[data-testid="stMetric"] {
          border: 1px solid #d8dee7; border-left: 3px solid #0f766e;
          border-radius: 6px; padding: 0.7rem 0.85rem; background: #ffffff;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.78rem; color: #475569;}
        .stButton > button, .stDownloadButton > button {border-radius: 6px;}
        div[data-baseweb="tab-list"] {gap: 1.1rem;}
        div[data-baseweb="tab"] {height: 2.8rem; padding-left: 0.25rem; padding-right: 0.25rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _provider_details() -> tuple[str, str, bool]:
    load_dotenv(PROJECT_ROOT / ".env")
    configured_model = os.getenv("LLM_MODEL")
    if os.getenv("GEMINI_API_KEY"):
        return "Gemini", configured_model or "gemini-2.5-flash", True
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI", configured_model or "gpt-4o-mini", True
    if os.getenv("ANTHROPIC_API_KEY"):
        return "Anthropic", configured_model or "claude-sonnet-4", True
    return "Not configured", "-", False


@st.cache_resource(show_spinner=False)
def _compiled_graph() -> CompiledStateGraph:
    return build_graph(build_checkpointer("memory"))


def _load_existing_metrics() -> MetricsReport | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return MetricsReport.model_validate_json(METRICS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _event_rows(state: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for index, event in enumerate(state.get("events", []), start=1):
        rows.append(
            {
                "Step": index,
                "Node": event.get("node", "unknown"),
                "Event": event.get("event_type", ""),
                "Latency (ms)": event.get("latency_ms", 0),
                "Message": event.get("message", ""),
            }
        )
    return rows


def _render_run_summary(
    graph: CompiledStateGraph,
    state: dict[str, Any],
    config: RunnableConfig,
) -> None:
    pending = interrupt_payload(state)
    events = state.get("events", [])
    retries = sum(1 for event in events if event.get("node") == "retry")
    status = "Approval required" if pending else "Completed"

    metric_columns = st.columns(4)
    metric_columns[0].metric("Route", state.get("route") or "pending")
    metric_columns[1].metric("Status", status)
    metric_columns[2].metric("Retries", retries)
    metric_columns[3].metric("Nodes visited", len(events))

    if pending:
        st.warning(str(pending.get("proposed_action", "Approval is required.")))
        comment = st.text_input("Reviewer comment", key="reviewer_comment")
        approve_column, reject_column, _ = st.columns([1, 1, 4])
        if approve_column.button("Approve", type="primary", use_container_width=True):
            with st.spinner("Resuming approved action..."):
                st.session_state.ticket_state = resume_ticket(graph, config, True, comment)
            st.rerun()
        if reject_column.button("Reject", use_container_width=True):
            with st.spinner("Recording rejection..."):
                st.session_state.ticket_state = resume_ticket(graph, config, False, comment)
            st.rerun()
    elif state.get("pending_question"):
        st.info(state["pending_question"])
    elif state.get("final_answer"):
        st.success(state["final_answer"])

    event_tab, checkpoint_tab, state_tab = st.tabs(["Event trace", "Checkpoints", "State"])
    with event_tab:
        st.dataframe(_event_rows(state), use_container_width=True, hide_index=True)
    with checkpoint_tab:
        snapshots = list(graph.get_state_history(config))
        snapshot_rows = [
            {
                "Checkpoint": index,
                "Next": ", ".join(snapshot.next) or "END",
                "Events": len(snapshot.values.get("events", [])),
                "Step": (snapshot.metadata or {}).get("step", ""),
            }
            for index, snapshot in enumerate(snapshots, start=1)
        ]
        st.dataframe(snapshot_rows, use_container_width=True, hide_index=True)
    with state_tab:
        serializable_state = {key: value for key, value in state.items() if key != "__interrupt__"}
        st.json(serializable_state)


def _render_agent(
    graph: CompiledStateGraph,
    provider_ready: bool,
    max_attempts: int,
    pause_risky: bool,
) -> None:
    sample = st.selectbox("Sample ticket", list(SAMPLE_QUERIES), index=0) or "Password help"
    query = st.text_area(
        "Ticket",
        value=SAMPLE_QUERIES[sample],
        height=110,
        key=f"ticket_query_{sample}",
    )
    run_column, clear_column, _ = st.columns([1, 1, 5])
    if run_column.button(
        "Run ticket",
        type="primary",
        disabled=not provider_ready,
        use_container_width=True,
    ):
        if not query.strip():
            st.warning("Ticket cannot be empty.")
        else:
            with st.spinner("Running graph..."):
                result, run_config = invoke_ticket(
                    graph,
                    query,
                    max_attempts=max_attempts,
                    interrupt_enabled=pause_risky,
                )
                st.session_state.ticket_state = result
                st.session_state.ticket_config = run_config
    if clear_column.button("Clear", use_container_width=True):
        st.session_state.pop("ticket_state", None)
        st.session_state.pop("ticket_config", None)
        st.rerun()

    state = cast(dict[str, Any] | None, st.session_state.get("ticket_state"))
    config = cast(RunnableConfig | None, st.session_state.get("ticket_config"))
    if state is not None and config is not None:
        st.divider()
        _render_run_summary(graph, state, config)


def _suite_rows(report: MetricsReport) -> list[dict[str, object]]:
    return [
        {
            "Scenario": item.scenario_id,
            "Expected": item.expected_route,
            "Actual": item.actual_route,
            "Success": item.success,
            "Nodes": item.nodes_visited,
            "Retries": item.retry_count,
            "Approvals": item.interrupt_count,
            "Latency (ms)": item.latency_ms,
        }
        for item in report.scenario_metrics
    ]


def _render_scenarios(
    graph: CompiledStateGraph,
    provider_ready: bool,
) -> None:
    if st.button("Run lab suite", type="primary", disabled=not provider_ready):
        scenarios = load_scenarios(SCENARIOS_PATH)
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def update_progress(completed: int, total: int, scenario: object) -> None:
            scenario_id = getattr(scenario, "id", "scenario")
            progress_bar.progress(completed / total)
            progress_text.caption(f"{completed}/{total} | {scenario_id}")

        with st.spinner("Running grading scenarios..."):
            report, _ = run_scenario_suite(
                graph,
                scenarios,
                persistent_backend=False,
                progress=update_progress,
            )
            write_metrics(report, METRICS_PATH)
            write_report(report, REPORT_PATH)
            st.session_state.suite_report = report
        progress_text.empty()

    display_report = cast(
        MetricsReport | None,
        st.session_state.get("suite_report") or _load_existing_metrics(),
    )
    if display_report is None:
        return

    summary_columns = st.columns(5)
    summary_columns[0].metric("Scenarios", display_report.total_scenarios)
    summary_columns[1].metric("Success", f"{display_report.success_rate:.0%}")
    summary_columns[2].metric("Avg nodes", f"{display_report.avg_nodes_visited:.2f}")
    summary_columns[3].metric("Retries", display_report.total_retries)
    summary_columns[4].metric("Approvals", display_report.total_interrupts)

    if display_report.resume_success:
        st.success("Checkpoint recovery verified.")
    st.dataframe(_suite_rows(display_report), use_container_width=True, hide_index=True)

    metrics_json = json.dumps(display_report.model_dump(), indent=2, ensure_ascii=False)
    report_markdown = render_report(display_report)
    metrics_column, report_column, _ = st.columns([1, 1, 4])
    metrics_column.download_button(
        "Download metrics",
        metrics_json,
        file_name="metrics.json",
        mime="application/json",
        use_container_width=True,
    )
    report_column.download_button(
        "Download report",
        report_markdown,
        file_name="lab_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_architecture() -> None:
    st.graphviz_chart(GRAPH_DOT, use_container_width=True)
    st.subheader("State contract")
    st.dataframe(
        [
            {"Field": "route, risk_level", "Reducer": "overwrite", "Owner": "classify"},
            {"Field": "attempt, max_attempts", "Reducer": "overwrite", "Owner": "retry"},
            {"Field": "evaluation_result", "Reducer": "overwrite", "Owner": "evaluate"},
            {
                "Field": "pending_question, proposed_action, approval",
                "Reducer": "overwrite",
                "Owner": "clarify / approval",
            },
            {
                "Field": "messages, tool_results, errors, events",
                "Reducer": "append",
                "Owner": "workflow audit",
            },
            {"Field": "final_answer", "Reducer": "overwrite", "Owner": "terminal nodes"},
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_report() -> None:
    report = st.session_state.get("suite_report") or _load_existing_metrics()
    if report is None:
        st.info("No report is available.")
        return
    markdown = render_report(report)
    st.download_button(
        "Download Markdown",
        markdown,
        file_name="lab_report.md",
        mime="text/markdown",
    )
    st.markdown(markdown)


def main() -> None:
    _configure_page()
    provider, model, provider_ready = _provider_details()

    st.title("LangGraph Support Console")
    st.caption(f"{provider} | {model}")

    with st.sidebar:
        st.subheader("Runtime")
        st.caption("Checkpoint: Memory (demo session)")
        max_attempts = int(st.number_input("Max attempts", min_value=1, max_value=8, value=3))
        pause_risky = st.checkbox("Pause risky actions", value=True)
        st.divider()
        st.caption(f"Provider: {provider}")
        st.caption(f"Model: {model}")
        if not provider_ready:
            st.error("LLM API key is not configured.")
        if st.button("New session", use_container_width=True):
            for key in ("ticket_state", "ticket_config", "reviewer_comment"):
                st.session_state.pop(key, None)
            st.rerun()

    graph = _compiled_graph()
    agent_tab, scenarios_tab, architecture_tab, report_tab = st.tabs(
        ["Agent", "Scenarios", "Architecture", "Report"]
    )
    with agent_tab:
        _render_agent(graph, provider_ready, max_attempts, pause_risky)
    with scenarios_tab:
        _render_scenarios(graph, provider_ready)
    with architecture_tab:
        _render_architecture()
    with report_tab:
        _render_report()


if __name__ == "__main__":
    main()
