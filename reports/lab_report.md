# Day 08 Lab Report

## 1. Team / student

- Name: Not provided
- Repo/commit: Working tree
- Date: 2026-08-25

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
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total approval events | 2 |
| Checkpoint recovery verified | no |

## 5. Scenario results

| Scenario | Expected | Actual | Success | Retries | Approvals | Latency ms | Errors |
|---|---|---|---:|---:|---:|---:|---|
| S01_simple | simple | simple | yes | 0 | 0 | 4686 | - |
| S02_tool | tool | tool | yes | 0 | 0 | 2767 | - |
| S03_missing | missing_info | missing_info | yes | 0 | 0 | 917 | - |
| S04_risky | risky | risky | yes | 0 | 1 | 3068 | - |
| S05_error | error | error | yes | 2 | 0 | 4915 | Transient failure recorded before attempt 1; Transient failure recorded before attempt 2 |
| S06_delete | risky | risky | yes | 0 | 1 | 2765 | - |
| S07_dead_letter | error | error | yes | 1 | 0 | 920 | Transient failure recorded before attempt 1 |

## 6. Failure analysis

1. A transient tool failure returns an explicit error result. The evaluate node gates it
   into retry, while `attempt >= max_attempts` prevents an unbounded loop and escalates
   the request to dead letter.
2. A side-effecting request cannot reach the tool directly. It first records a proposed
   action and approval decision; rejection routes to clarification without execution.
3. Structured LLM output prevents free-form route parsing. Tool evaluation retains a
   deterministic fallback if the judge call is temporarily unavailable.

## 7. Persistence and recovery evidence

Each run uses a unique `thread_id` and a LangGraph checkpointer. SQLite runs use WAL mode,
and the scenario runner reads the saved state back after invocation to verify recovery.

## 8. Extension work

The SQLite extension stores graph checkpoints in a file-backed database with WAL mode.
The recovery check rebuilds a graph with the same checkpoint database and reads state by
`thread_id`. Real LangGraph interrupts are also available through `LANGGRAPH_INTERRUPT`;
batch grading explicitly uses mock approval so unattended runs terminate.
The Streamlit console demonstrates single-ticket execution, approve/reject resume,
checkpoint history and recovery, batch metrics, architecture, and report rendering.

## 9. Improvement plan

Production work should replace the mock tool and mock reviewer with authenticated APIs,
resume real HITL interrupts from an operator UI, redact sensitive event data, and add
tracing plus provider-level timeout and rate-limit handling.
