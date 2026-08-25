"""Persistence extension tests."""

import pytest

from langgraph_agent_lab.persistence import build_checkpointer, resolve_checkpointer_config


def test_environment_overrides_checkpointer_config(monkeypatch):
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/lab")

    kind, database_url = resolve_checkpointer_config(
        {"checkpointer": "memory"}
    )

    assert kind == "postgres"
    assert database_url == "postgresql://example.invalid/lab"


def test_unknown_checkpointer_is_rejected():
    with pytest.raises(ValueError, match="Unknown checkpointer kind"):
        build_checkpointer("unknown")


def test_postgres_requires_database_url():
    with pytest.raises(ValueError, match="database_url is required"):
        build_checkpointer("postgres")
