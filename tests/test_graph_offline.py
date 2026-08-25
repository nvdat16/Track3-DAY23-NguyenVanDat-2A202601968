"""End-to-end graph tests that do not require a provider API key."""

from dataclasses import dataclass

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.nodes import ClassificationDecision, EvaluationDecision
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.runner import interrupt_payload, invoke_ticket, resume_ticket
from langgraph_agent_lab.state import Route, Scenario, initial_state


@dataclass
class FakeMessage:
    content: str


class FakeStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        if self.schema is EvaluationDecision:
            tool_result = prompt.lower().rsplit("tool result:", 1)[-1]
            result = "needs_retry" if "status: error" in tool_result else "success"
            return EvaluationDecision(evaluation_result=result, reason="fake judge")

        normalized = prompt.lower().rsplit("request:", 1)[-1]
        if "refund" in normalized or "delete" in normalized:
            route = "risky"
        elif "lookup" in normalized:
            route = "tool"
        elif "fix it" in normalized:
            route = "missing_info"
        elif "timeout" in normalized or "failure" in normalized:
            route = "error"
        else:
            route = "simple"
        return ClassificationDecision(route=route, reason="fake classifier")


class FakeChatModel:
    def with_structured_output(self, schema):
        return FakeStructuredModel(schema)

    def invoke(self, prompt):
        return FakeMessage(content="Grounded test answer")


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE),
        ("Please lookup order 123", Route.TOOL),
        ("Can you fix it?", Route.MISSING_INFO),
        ("Refund this customer", Route.RISKY),
        ("Timeout failure while processing", Route.ERROR),
    ],
)
def test_all_routes_terminate(monkeypatch, query, expected_route):
    monkeypatch.setattr("langgraph_agent_lab.nodes.get_llm", lambda **_: FakeChatModel())
    graph = build_graph(build_checkpointer("memory"))
    scenario = Scenario(id=expected_route.value, query=query, expected_route=expected_route)
    state = initial_state(scenario)

    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    assert result["route"] == expected_route.value
    assert result.get("final_answer") or result.get("pending_question")
    assert any(event["node"] == "finalize" for event in result["events"])


def test_error_route_reaches_dead_letter_at_limit(monkeypatch):
    monkeypatch.setattr("langgraph_agent_lab.nodes.get_llm", lambda **_: FakeChatModel())
    graph = build_graph(build_checkpointer("memory"))
    scenario = Scenario(
        id="dead-letter",
        query="System failure cannot recover",
        expected_route=Route.ERROR,
        max_attempts=1,
    )
    state = initial_state(scenario)

    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    nodes = [event["node"] for event in result["events"]]
    assert nodes == ["intake", "classify", "retry", "dead_letter", "finalize"]


def test_risky_ticket_interrupts_and_resumes(monkeypatch):
    monkeypatch.setattr("langgraph_agent_lab.nodes.get_llm", lambda **_: FakeChatModel())
    graph = build_graph(build_checkpointer("memory"))

    paused, config = invoke_ticket(graph, "Refund this customer", interrupt_enabled=True)

    assert interrupt_payload(paused) is not None
    assert graph.get_state(config).next == ("approval",)

    resumed = resume_ticket(graph, config, approved=False, comment="Use another option")
    nodes = [event["node"] for event in resumed["events"]]
    assert resumed["approval"]["approved"] is False
    assert resumed["pending_question"]
    assert nodes[-2:] == ["clarify", "finalize"]
