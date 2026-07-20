"""Test the interactive `run` orchestrator via its non-interactive `--yes` path (S20)."""

from __future__ import annotations

from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.generate import make_panel

runner = CliRunner()


def _write_config(tmp_path) -> tuple:
    panel = tmp_path / "panel.parquet"
    make_panel(n_accounts=600, n_months=8, seed=3).to_parquet(panel, index=False)
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        f"source:\n  kind: file\n  path: {panel}\n"
        "schema:\n"
        "  id_col: account_id\n"
        "  target_col: churn_next_30d\n"
        "  date_col: observation_month\n"
        "  value_col: cltv\n"
        "  features: auto\n"
    )
    return cfg, tmp_path / "run"


def test_run_yes_end_to_end(tmp_path):
    cfg, out = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(cfg),
            "--yes",
            "--models",
            "logistic,rf",
            "--out-dir",
            str(out),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Every artifact + the report land in the run directory.
    for f in (
        "model.pkl",
        "model.card.json",
        "eval-report.json",
        "policy-report.json",
        "report.html",
    ):
        assert (out / f).exists(), f"missing {f}"


def test_run_yes_walks_every_gate_and_drops_the_leak(tmp_path):
    cfg, out = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["run", "--config", str(cfg), "--yes", "--models", "logistic,rf", "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    text = result.output
    # Each decision gate is surfaced...
    for gate in ("features", "split", "model", "ship", "policy"):
        assert f"[{gate}]" in text
    # ...and --yes takes the recommendation to exclude the planted leak feature.
    assert "cancel_page_visits_30d" in text and "Exclude" in text
    assert "✔ done" in text


def test_run_report_is_self_contained(tmp_path):
    cfg, out = _write_config(tmp_path)
    runner.invoke(
        app, ["run", "--config", str(cfg), "--yes", "--models", "logistic", "--out-dir", str(out)]
    )
    html = (out / "report.html").read_text()
    assert html.startswith("<!doctype html>") and "data:image/png;base64," in html
