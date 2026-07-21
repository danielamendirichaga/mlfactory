"""Tests for the EDA deterministic substrate — scan_leakage tiers + the eda-exploration artifact."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from mlfactory.artifacts.schemas import EdaExplorationArtifact
from mlfactory.artifacts.validate import validate_artifact
from mlfactory.cli import app
from mlfactory.compute.profile import profile_frame, scan_leakage
from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel

runner = CliRunner()

SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
}


def _cfg(schema=SCHEMA):
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


def test_scan_flags_the_planted_leak_not_the_drivers():
    df = make_panel(n_accounts=800, n_months=10, seed=3)
    by_col = {r["column"]: r for r in scan_leakage(profile_frame(df, _cfg()), _cfg())}
    assert "cancel_page_visits_30d" in by_col
    assert by_col["cancel_page_visits_30d"]["kind"] in ("perfect_predictor", "near_perfect")
    assert by_col["cancel_page_visits_30d"]["recommendation"] in ("drop", "inspect")
    # genuine drivers do NOT cross the 0.9 leakage tier
    for driver in ("product_usage_hours_30d", "tenure_months", "days_since_last_login"):
        assert driver not in by_col


def test_scan_tiers_by_correlation():
    n = 200
    y = np.array([0, 1] * (n // 2))
    df = pd.DataFrame(
        {
            "account_id": range(n),
            "churn_next_30d": y,
            "leak": y.astype(float),  # corr 1.0 → perfect_predictor
            "weak": y * 0.1 + np.random.default_rng(0).normal(0, 1, n),  # weak → not flagged
        }
    )
    cfg = _cfg({"id_col": "account_id", "target_col": "churn_next_30d"})
    risks = {r["column"]: r for r in scan_leakage(profile_frame(df, cfg), cfg)}
    assert (
        risks["leak"]["kind"] == "perfect_predictor" and risks["leak"]["recommendation"] == "drop"
    )
    assert "weak" not in risks


def test_cli_leakage_scan_json(tmp_path):
    panel = tmp_path / "p.parquet"
    make_panel(n_accounts=400, n_months=8, seed=5).to_parquet(panel, index=False)
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        f"source:\n  kind: file\n  path: {panel}\n"
        "schema:\n  id_col: account_id\n  target_col: churn_next_30d\n  date_col: observation_month\n"
    )
    r = runner.invoke(app, ["leakage-scan", "--config", str(cfg), "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip())
    assert payload["command"] == "leakage-scan"
    assert any(x["column"] == "cancel_page_visits_30d" for x in payload["leakage_risks"])


def test_eda_exploration_artifact_validates_and_roundtrips(tmp_path):
    art = EdaExplorationArtifact(
        mode="modeling",
        targets=["churn_next_30d"],
        feature_candidates=["tenure_months", "product_usage_hours_30d"],
        leakage_risks=[
            {
                "column": "cancel_page_visits_30d",
                "target": "churn_next_30d",
                "strength": 0.92,
                "kind": "posterior_info",
                "recommendation": "drop",
                "reason": "posterior cancel-flow signal, not observable at prediction time",
            }
        ],
        recommended_model_families=[
            {"family": "logistic", "rank": 1, "reasoning": "n<5k → regularized linear most stable"}
        ],
        baseline_spec={"type": "logreg_3feat", "metric_name": "auc"},
        cv_strategy={"type": "time_aware"},
    )
    p = tmp_path / "eda.md"
    art.write_markdown(p)
    assert validate_artifact(p)["artifact"] == "eda-exploration"
    assert EdaExplorationArtifact.from_markdown(p.read_text()) == art
