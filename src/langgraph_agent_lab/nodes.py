"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Literal, cast

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event


class ClassificationDecision(BaseModel):
    """Structured classification returned by the language model."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reason: str = Field(description="Brief reason for selecting the route")


class EvaluationDecision(BaseModel):
    """Structured quality decision for a tool result."""

    evaluation_result: Literal["success", "needs_retry"]
    reason: str = Field(description="Brief reason for the quality decision")


def _latency_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _response_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# Workflow nodes


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    started_at = perf_counter()
    prompt = f"""You route support requests into exactly one workflow.

Routes:
- risky: requests that cause side effects, including refunds, deletion, cancellation,
  account changes, or sending messages.
- tool: information lookup or search that requires an external system.
- missing_info: vague or incomplete requests without enough context to act.
- error: reports of a system timeout, crash, outage, or processing failure.
- simple: general guidance answerable without tools or side effects.

When multiple routes apply, use this priority: risky, tool, missing_info, error, simple.
Classify by intent rather than matching examples or identifiers.

Request: {state.get("query", "")}
"""
    classifier = get_llm(temperature=0.0).with_structured_output(ClassificationDecision)
    decision = cast(ClassificationDecision, classifier.invoke(prompt))
    route = decision.route
    return {
        "route": route,
        "risk_level": "high" if route == Route.RISKY.value else "low",
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                latency_ms=_latency_ms(started_at),
                reason=decision.reason,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    started_at = perf_counter()
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    if route == Route.ERROR.value and attempt < 2:
        result = f"STATUS: ERROR. Transient support-system failure on attempt {attempt}."
        event_type = "failed"
    elif route == Route.RISKY.value:
        result = "STATUS: SUCCESS. The approved support action was executed."
        event_type = "completed"
    else:
        result = "STATUS: SUCCESS. The support-system lookup completed."
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                event_type,
                "mock tool call finished",
                latency_ms=_latency_ms(started_at),
                attempt=attempt,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    started_at = perf_counter()
    latest_result = (state.get("tool_results") or [""])[-1]
    prompt = f"""Judge whether this support-tool result can be used to answer the user.
Treat an explicit `STATUS: ERROR` as needs_retry and an explicit `STATUS: SUCCESS`
as success. Otherwise, return needs_retry for unavailable or incomplete execution.

Tool result: {latest_result}
"""
    try:
        judge = get_llm(temperature=0.0).with_structured_output(EvaluationDecision)
        decision = cast(EvaluationDecision, judge.invoke(prompt))
        evaluation_result = decision.evaluation_result
        reason = decision.reason
        event_type = "completed"
    except Exception as exc:
        evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"
        reason = f"heuristic fallback after judge error: {type(exc).__name__}"
        event_type = "fallback"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                event_type,
                f"tool result evaluated as {evaluation_result}",
                latency_ms=_latency_ms(started_at),
                reason=reason,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    started_at = perf_counter()
    tool_context = "\n".join(state.get("tool_results") or []) or "No tool was required."
    approval = state.get("approval")
    prompt = f"""You are a concise, helpful support assistant.
Answer the user's request using only the supplied context. Do not invent account,
order, or execution details. If an approved action was executed, confirm it clearly.

User request: {state.get("query", "")}
Tool context: {tool_context}
Approval context: {approval or "Not required"}
"""
    response = get_llm(temperature=0.1).invoke(prompt)
    final_answer = _response_text(response)
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "grounded answer generated",
                latency_ms=_latency_ms(started_at),
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "your request")
    question = (
        "Please provide the affected account, order, or system component and describe "
        f"the expected outcome for this request: {query}"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "waiting_for_user", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = (
        f"Execute the requested side-effecting support action: {state.get('query', '')}. "
        "Human approval is required before any external change is made."
    )
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "prepared", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval decision and its audit event.
    """
    interrupt_enabled = state.get("interrupt_enabled")
    if interrupt_enabled is None:
        interrupt_enabled = os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true"

    if interrupt_enabled:
        from langgraph.types import interrupt

        response = interrupt({"proposed_action": state.get("proposed_action", "")})
        if isinstance(response, dict):
            decision = ApprovalDecision.model_validate(response)
        else:
            decision = ApprovalDecision(approved=bool(response), reviewer="human-reviewer")
        event_type = "human_decision"
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for deterministic lab execution.",
        )
        event_type = "mock_decision"
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                event_type,
                "action approved" if decision.approved else "action rejected",
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    error = f"Transient failure recorded before attempt {attempt}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [
            make_event(
                "retry",
                "scheduled",
                "retry attempt recorded",
                attempt=attempt,
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = (
        "The request could not be completed after the allowed retry attempts. "
        "It has been recorded for manual support follow-up."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "escalated", "retry limit exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
