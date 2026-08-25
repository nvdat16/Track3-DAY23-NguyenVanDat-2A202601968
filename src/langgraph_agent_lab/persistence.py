"""Checkpointer adapter."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver


def resolve_checkpointer_config(config: Mapping[str, Any]) -> tuple[str, str | None]:
    """Resolve persistence settings, preferring values from the environment."""
    kind = os.getenv("CHECKPOINTER") or str(config.get("checkpointer", "memory"))
    configured_url = config.get("database_url")
    database_url = os.getenv("DATABASE_URL") or (
        str(configured_url) if configured_url else None
    )
    return kind.strip().lower(), database_url


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    SQLite connections remain owned by the saver for the graph lifetime. File-backed
    databases use WAL mode so checkpoints survive process restarts safely.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the project with the 'sqlite' extra") from exc

        location = database_url or "outputs/checkpoints.sqlite"
        if location.startswith("sqlite:///"):
            location = location.removeprefix("sqlite:///")
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(location, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return SqliteSaver(conn=connection)
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
