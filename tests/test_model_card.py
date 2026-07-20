"""Tests for the model-card renderer and the --json CLI tool surface."""

from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.domains.saas.generate import make_panel
from mlfactory.model_card import SECTIONS, gen_model_card

runner = CliRunner()

_MC = {
    "artifact": "model-card",
    "version": "1.0",
    "model_family": "logistic",
    "tuned": False,
    "smote": False,
    "calibrated": True,
    "early_stopping": False,
    "n_features": 3,
    "features": ["tenure_months", "mrr", "plan_tier"],
    "hyperparams": {"C": 1.0},
    "train_metrics": {"auc": 0.72, "ks": 0.3, "top_decile_lift": 2.1},
    "baseline_metrics": {"auc": 0.5},
    "parent_sha256": "a" * 64,
}
_EV = {
    "n_rows": 500,
    "metrics": {
        "auc": 0.68,
        "pr_auc": 0.4,
        "ks": 0.28,
        "top_decile_lift": 1.9,
        "log_loss": 0.3,
        "ece": 0.02,
    },
    "segments": {"plan_tier": {"Starter": {"n": 300, "churn_rate": 0.1, "auc": 0.66, "lift": 1.8}}},
}


# --- renderer ------------------------------------------------------------- #
def test_gen_model_card_has_all_sections():
    md = gen_model_card(_MC, _EV, target="churn_next_30d")
    for section in SECTIONS:
        assert f"## {section}" in md
    assert "logistic" in md
    assert "0.6800" in md  # test AUC
    assert "0.5000" in md  # baseline floor
    assert "tenure_months" in md
    assert "0.0200" in md  # ECE
    assert "Starter" in md  # a slice row


def test_gen_model_card_without_eval_omits_calibration_and_slices():
    md = gen_model_card(_MC, None)
    assert "## Purpose" in md and "## Lineage" in md
    assert "## Calibration" not in md and "## Slices" not in md


def test_cli_gen_model_card(tmp_path):
    (tmp_path / "model.card.json").write_text(json.dumps(_MC))
    (tmp_path / "eval-report.json").write_text(json.dumps(_EV))
    out = tmp_path / "card.md"
    r = runner.invoke(
        app,
        [
            "gen-model-card",
            "--card",
            str(tmp_path / "model.card.json"),
            "--eval",
            str(tmp_path / "eval-report.json"),
            "--output",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    text = out.read_text()
    assert text.startswith("# Model Card") and "## Performance" in text


# --- the --json tool surface (what subagents will parse) ------------------ #
def _write_cfg(tmp_path):
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        "source:\n  kind: synthetic\n"
        "schema:\n  id_col: account_id\n  target_col: churn_next_30d\n"
        "  date_col: observation_month\n  features: [tenure_months, mrr, plan_tier]\n"
    )
    return cfg


def test_train_json_output(tmp_path):
    panel = tmp_path / "train.parquet"
    make_panel(n_accounts=300, n_months=8, seed=2).to_parquet(panel, index=False)
    r = runner.invoke(
        app,
        [
            "train",
            "--train",
            str(panel),
            "--config",
            str(_write_cfg(tmp_path)),
            "--model",
            "logistic",
            "--model-out",
            str(tmp_path / "m.pkl"),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip())
    assert payload["command"] == "train" and payload["model"] == "logistic"
    assert "auc" in payload["train_metrics"] and "auc" in payload["baseline_metrics"]


def test_engineer_features_json_output(tmp_path):
    make_panel(n_accounts=200, n_months=6, seed=4).to_parquet(
        tmp_path / "train.parquet", index=False
    )
    (tmp_path / "spec.yaml").write_text(
        yaml.safe_dump(
            {
                "transforms": [
                    {
                        "id": 1,
                        "name": "scale",
                        "type": "standard_scaler",
                        "inputs": ["tenure_months"],
                        "output_column": "ts",
                    }
                ]
            }
        )
    )
    r = runner.invoke(
        app,
        [
            "engineer-features",
            "--train",
            str(tmp_path / "train.parquet"),
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip())
    assert payload["command"] == "engineer-features" and "ts" in payload["produced"]
