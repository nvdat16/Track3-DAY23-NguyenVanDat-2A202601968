"""Persistence extension tests."""

import importlib.util

import pytest

from langgraph_agent_lab.persistence import build_checkpointer, resolve_checkpointer_config


def test_environment_overrides_checkpointer_config(monkeypatch):
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/lab")

    kind, database_url = resolve_checkpointer_config(
        {"checkpointer": "sqlite", "database_url": "outputs/local.sqlite"}
    )

    assert kind == "postgres"
    assert database_url == "postgresql://example.invalid/lab"


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="SQLite checkpoint extra is not installed",
)
def test_sqlite_checkpointer_uses_wal(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"
    saver = build_checkpointer("sqlite", str(database_path))

    assert saver is not None
    journal_mode = saver.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode == "wal"
    saver.conn.close()


def test_unknown_checkpointer_is_rejected():
    with pytest.raises(ValueError, match="Unknown checkpointer kind"):
        build_checkpointer("unknown")


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="SQLite checkpoint extra is not installed",
)
def test_sqlite_recovery_uses_new_graph(monkeypatch, tmp_path):
    from test_graph_offline import FakeChatModel

    from langgraph_agent_lab import nodes
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.runner import invoke_ticket, verify_checkpoint_recovery

    monkeypatch.setattr(nodes, "get_llm", lambda **_: FakeChatModel())
    database_path = tmp_path / "recovery.sqlite"
    saver = build_checkpointer("sqlite", str(database_path))
    graph = build_graph(saver)
    _, config = invoke_ticket(graph, "How do I reset my password?", interrupt_enabled=False)

    assert verify_checkpoint_recovery("sqlite", str(database_path), [config]) is True
    saver.conn.close()
