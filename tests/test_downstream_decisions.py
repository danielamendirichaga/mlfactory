"""Slice S6 — the downstream stages read the decision record.

`simulate-policy` / `policy-contrast` take `save_rate` / `offer_cost` / `budget` from
`config.decisions.policy`, and `monitor` takes the drift bar from `config.decisions.monitoring` —
instead of the silent `0.3` / `$5` / `none` / `0.25` defaults. CLI flags still override.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.config import DecisionRecord

runner = CliRunner()


def test_policy_and_monitoring_defaults() -> None:
    d = DecisionRecord()
    assert (d.policy.save_rate, d.policy.offer_cost, d.policy.budget, d.policy.targeting) == (
        0.3,
        5.0,
        None,
        "risk",
    )
    assert d.monitoring.drift_threshold == 0.25


def test_monitor_reads_the_drift_bar_from_the_record(tmp_path: Path) -> None:
    cfg = str(tmp_path / "churn.yaml")
    assert runner.invoke(app, ["init", "--path", cfg]).exit_code == 0
    runner.invoke(
        app,
        [
            "record-decision",
            "--config",
            cfg,
            "--key",
            "monitoring.drift_threshold",
            "--value",
            "0.4",
        ],
    )
    out = tmp_path / "drift.json"
    result = runner.invoke(app, ["monitor", "--config", cfg, "--report-out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["threshold"] == 0.4


def test_simulate_policy_reads_economics_from_the_record(tmp_path: Path) -> None:
    cfg = str(tmp_path / "churn.yaml")
    data = str(tmp_path / "panel.parquet")
    model = str(tmp_path / "m.pkl")
    assert runner.invoke(app, ["init", "--path", cfg]).exit_code == 0
    assert (
        runner.invoke(
            app, ["generate", "--out", data, "--accounts", "300", "--months", "6"]
        ).exit_code
        == 0
    )
    runner.invoke(app, ["exclude-columns", "--config", cfg, "--add", "cancel_page_visits_30d"])
    train = runner.invoke(
        app,
        ["train", "--train", data, "--config", cfg, "--model", "logistic", "--model-out", model],
    )
    assert train.exit_code == 0, train.output
    runner.invoke(
        app, ["record-decision", "--config", cfg, "--key", "policy.offer_cost", "--value", "9"]
    )
    report = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "simulate-policy",
            "--model",
            model,
            "--data",
            data,
            "--config",
            cfg,
            "--report-out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        json.loads(report.read_text())["offer_cost"] == 9.0
    )  # from the record, no --offer-cost flag
