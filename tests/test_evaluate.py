"""Tests for held-out evaluation (S9) — the union metric pack + segments + eval-report."""

from __future__ import annotations

import pytest

from mlfactory.config import ChurnConfig
from mlfactory.evaluate import EvalReport, evaluate_model
from mlfactory.generate import make_panel
from mlfactory.model import train_model

FEATURES = [
    "tenure_months",
    "mrr",
    "product_usage_hours_30d",
    "active_days_30d",
    "days_since_last_login",
    "usage_trend_30d",
    "support_tickets_30d",
    "plan_tier",
    "region",
]
SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
    "features": FEATURES,
}


def _cfg():
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SCHEMA})


@pytest.fixture(scope="module")
def fitted():
    train = make_panel(n_accounts=1200, n_months=10, seed=31)
    test = make_panel(n_accounts=500, n_months=10, seed=32)
    est, _ = train_model(train, _cfg(), model="logistic", seed=1)
    return est, train, test


def test_report_has_full_metric_pack(fitted):
    est, _, test = fitted
    report = evaluate_model(est, test, _cfg())
    assert {
        "auc",
        "pr_auc",
        "ks",
        "rank_order_breaks",
        "top_decile_lift",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "ece",
    } <= set(report.metrics)
    assert 0.4 < report.metrics["auc"] < 1.0
    assert report.metrics["ece"] >= 0
    assert report.calibration  # reliability table present


def test_per_segment_slices(fitted):
    est, _, test = fitted
    report = evaluate_model(est, test, _cfg())
    assert set(report.segments) == {"plan_tier", "region"}
    for seg in report.segments.values():
        for level in seg.values():
            assert level["n"] > 0
            assert 0.0 <= level["churn_rate"] <= 1.0
            assert level["auc"] is None or 0.0 <= level["auc"] <= 1.0


def test_score_psi_when_reference_given(fitted):
    est, train, test = fitted
    no_ref = evaluate_model(est, test, _cfg())
    assert no_ref.score_psi is None
    with_ref = evaluate_model(est, test, _cfg(), reference_df=train)
    assert with_ref.score_psi is not None and with_ref.score_psi >= 0


def test_artifact_lineage_and_roundtrip(fitted, tmp_path):
    est, _, test = fitted
    report = evaluate_model(est, test, _cfg())
    assert report.artifact == "eval-report"
    assert report.parent_sha256 and len(report.parent_sha256) == 64
    p = tmp_path / "eval-report.json"
    report.write_json(p)
    assert EvalReport.model_validate_json(p.read_text()) == report


def test_missing_feature_columns_raises(fitted):
    est, _, test = fitted
    with pytest.raises(ValueError, match="missing feature columns"):
        evaluate_model(est, test.drop(columns=["product_usage_hours_30d"]), _cfg())
