"""End-to-end CLI pipeline test — the deterministic happy path the /mlfactory-run orchestrator drives.

generate → split → engineer-features → validate-artifact → train → evaluate → gen-model-card, all via
the CLI (CliRunner). The automated guard behind the agent-layer foundation (#10): the pipeline the
playbook orchestrates actually runs green and produces a **validated feature-spec** + a **model card**.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mlfactory.cli import app

runner = CliRunner()

# the committed example the orchestrator playbook references
FEATURE_SPEC = Path(__file__).resolve().parents[1] / "docs" / "example-feature-spec.yaml"


def _invoke(args: list[str]):
    r = runner.invoke(app, args)
    assert r.exit_code == 0, f"`{args[0]}` failed:\n{r.output}"
    return r


def test_cli_pipeline_end_to_end(tmp_path):
    panel = tmp_path / "panel.parquet"
    _invoke(["generate", "--accounts", "500", "--months", "8", "--out", str(panel)])

    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        f"source:\n  kind: file\n  path: {panel}\n"
        "schema:\n  id_col: account_id\n  target_col: churn_next_30d\n"
        "  date_col: observation_month\n  value_col: cltv\n  features: auto\n"
    )

    # split (leakage-guarded, time-aware)
    splits = tmp_path / "splits"
    _invoke(["split", "--config", str(cfg), "--strategy", "time", "--out-dir", str(splits)])
    assert (splits / "train.parquet").exists() and (splits / "split-manifest.json").exists()

    # feature engineering → feature-spec artifact, gated by validate-artifact (schema + lineage + probe)
    feats = tmp_path / "features"
    _invoke(
        [
            "engineer-features",
            "--train",
            str(splits / "train.parquet"),
            "--val",
            str(splits / "val.parquet"),
            "--test",
            str(splits / "test.parquet"),
            "--spec",
            str(FEATURE_SPEC),
            "--output-dir",
            str(feats),
        ]
    )
    assert (feats / "feature-spec.md").exists()
    _invoke(
        ["validate-artifact", str(feats / "feature-spec.md"), "--walk-lineage", "--probe-output"]
    )

    # train → model + card · evaluate → eval-report · gen-model-card → the deliverable
    model = tmp_path / "model.pkl"
    _invoke(
        [
            "train",
            "--train",
            str(splits / "train.parquet"),
            "--config",
            str(cfg),
            "--model",
            "logistic",
            "--model-out",
            str(model),
        ]
    )
    ev = tmp_path / "eval-report.json"
    _invoke(
        [
            "evaluate",
            "--model",
            str(model),
            "--test",
            str(splits / "test.parquet"),
            "--config",
            str(cfg),
            "--report-out",
            str(ev),
        ]
    )
    card = tmp_path / "model-card.md"
    _invoke(
        [
            "gen-model-card",
            "--card",
            str(tmp_path / "model.card.json"),
            "--eval",
            str(ev),
            "--output",
            str(card),
        ]
    )
    text = card.read_text()
    assert text.startswith("# Model Card")
    assert "## Performance" in text and "## Lineage" in text
