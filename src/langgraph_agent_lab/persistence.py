"""Checkpointer adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver


def resolve_checkpointer_config(config: Mapping[str, Any]) -> tuple[str, str | None]:
    """Resolve persistence settings, preferring values from the environment."""
    kind = os.getenv("CHECKPOINTER") or str(config.get("checkpointer", "postgres"))
    configured_url = config.get("database_url")
    database_url = os.getenv("DATABASE_URL") or (
        str(configured_url) if configured_url else None
    )
    return kind.strip().lower(), database_url


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver | None:
    """Return a memory checkpointer for tests or PostgreSQL for persistent runs."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "postgres":
        if not database_url:
            raise ValueError("database_url is required for the postgres checkpointer")
        try:
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
                PostgresSaver,
            )
            from psycopg import Connection  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the project with the 'postgres' extra") from exc

        connection = Connection.connect(
            database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        saver = PostgresSaver(connection)
        saver.setup()
        return saver
    raise ValueError(f"Unknown checkpointer kind: {kind}")
