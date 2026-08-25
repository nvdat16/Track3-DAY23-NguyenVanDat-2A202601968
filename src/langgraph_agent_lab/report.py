"""Report generation helper."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Return: formatted markdown string
    """
    student_name = os.getenv("STUDENT_NAME", "Not provided")
    commit = os.getenv("GIT_COMMIT", "Working tree")
    scenario_rows = []
    for item in metrics.scenario_metrics:
        errors = "; ".join(item.errors).replace("|", "\\|") or "-"
        scenario_rows.append(
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} "
            f"| {'yes' if item.success else 'no'} | {item.retry_count} "
            f"| {item.interrupt_count} | {item.latency_ms} | {errors} |"
        )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: {student_name}
- Repo/commit: {commit}
- Date: {date.today().isoformat()}

## 2. Architecture

The workflow uses an append-only audit trail and explicit conditional edges:

`START -> intake -> classify -> route -> answer/tool/clarify/approval/retry -> finalize -> END`

Simple requests go directly to answer generation. Tool requests are evaluated and may
enter a bounded retry loop. Incomplete requests ask for clarification. Side-effecting
actions require approval before tool execution. Exhausted failures enter dead letter.

## 3. State schema

| Field | Reducer | Purpose |
|---|---|---|
| route, risk_level | overwrite | Current classification and risk decision |
| attempt, max_attempts | overwrite | Bound the retry loop |
| evaluation_result | overwrite | Gate answer versus retry |
| pending_question, proposed_action, approval | overwrite | Clarification and HITL state |
| messages, tool_results, errors, events | append | Serializable audit history |
| final_answer | overwrite | Terminal user-facing response |

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total approval events | {metrics.total_interrupts} |
| Checkpoint recovery verified | {'yes' if metrics.resume_success else 'no'} |

## 5. Scenario results

| Scenario | Expected | Actual | Success | Retries | Approvals | Latency ms | Errors |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(scenario_rows)}

## 6. Failure analysis

1. A transient tool failure returns an explicit error result. The evaluate node gates it
   into retry, while `attempt >= max_attempts` prevents an unbounded loop and escalates
   the request to dead letter.
2. A side-effecting request cannot reach the tool directly. It first records a proposed
   action and approval decision; rejection routes to clarification without execution.
3. Structured LLM output prevents free-form route parsing. Tool evaluation retains a
   deterministic fallback if the judge call is temporarily unavailable.

## 7. Persistence and recovery evidence

Each run uses a unique `thread_id` and a PostgreSQL LangGraph checkpointer. The scenario
runner opens a new database connection and reads saved state back by thread ID to verify
that checkpoints survive graph reconstruction.

## 8. Extension work

The PostgreSQL extension stores graph checkpoints in the Docker Compose database. The
recovery check rebuilds a graph with a new PostgreSQL connection and reads state by
`thread_id`. Real LangGraph interrupts are also available through `LANGGRAPH_INTERRUPT`;
batch grading explicitly uses mock approval so unattended runs terminate.
The separate Streamlit console demonstrates single-ticket execution, approve/reject
resume, in-session checkpoint history, batch metrics, architecture, and report rendering.

## 9. Improvement plan

Production work should replace the mock tool and mock reviewer with authenticated APIs,
resume real HITL interrupts from an operator UI, redact sensitive event data, and add
tracing plus provider-level timeout and rate-limit handling.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
