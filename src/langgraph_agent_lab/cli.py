"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

from .graph import build_graph
from .metrics import MetricsReport, write_metrics
from .persistence import build_checkpointer, resolve_checkpointer_config
from .report import write_report
from .runner import run_scenario_suite, verify_checkpoint_recovery
from .scenarios import load_scenarios

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    load_dotenv()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind, database_url = resolve_checkpointer_config(cfg)
    checkpointer = build_checkpointer(checkpointer_kind, database_url)
    graph = build_graph(checkpointer=checkpointer)
    persistent_backend = checkpointer_kind in {"sqlite", "postgres"}
    report, run_configs = run_scenario_suite(graph, scenarios, persistent_backend)
    if persistent_backend:
        report.resume_success = verify_checkpoint_recovery(
            checkpointer_kind,
            database_url,
            run_configs,
        )
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
