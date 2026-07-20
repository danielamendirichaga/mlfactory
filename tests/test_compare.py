"""Tests for model comparison (S8) — ranks the shortlist on performance + stability."""

from __future__ import annotations

import pytest

from mlfactory.compute.compare import compare_models
from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel

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


@pytest.fixture(scope="module")
def splits():
    train = make_panel(n_accounts=1200, n_months=10, seed=21)
    holdout = make_panel(n_accounts=500, n_months=10, seed=22)
    return train, holdout


def _cfg():
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SCHEMA})


def test_ranks_multiple_models(splits):
    train, holdout = splits
    rows = compare_models(train, holdout, _cfg(), models=["logistic", "rf", "xgboost"])
    assert len(rows) == 3
    # Sorted by held-out AUC, descending.
    aucs = [r["holdout_auc"] for r in rows]
    assert aucs == sorted(aucs, reverse=True)


def test_row_has_performance_and_stability(splits):
    train, holdout = splits
    row = compare_models(train, holdout, _cfg(), models=["logistic"])[0]
    assert {
        "model",
        "holdout_auc",
        "holdout_ks",
        "holdout_lift",
        "holdout_pr_auc",
        "auc_drop",
        "ks_drop",
        "score_psi",
        "stable",
    } <= set(row)
    assert row["score_psi"] >= 0
    assert isinstance(row["stable"], bool)


def test_captures_overfitting_gap(splits):
    """rf tends to overfit → a larger train→holdout AUC drop than L1 logistic."""
    train, holdout = splits
    rows = {
        r["model"]: r for r in compare_models(train, holdout, _cfg(), models=["logistic", "rf"])
    }
    assert rows["rf"]["auc_drop"] > rows["logistic"]["auc_drop"]


def test_single_model(splits):
    train, holdout = splits
    rows = compare_models(train, holdout, _cfg(), models=["logistic"])
    assert len(rows) == 1 and rows[0]["model"] == "logistic"


def test_deterministic(splits):
    train, holdout = splits
    a = compare_models(train, holdout, _cfg(), models=["logistic", "rf"], seed=7)
    b = compare_models(train, holdout, _cfg(), models=["logistic", "rf"], seed=7)
    assert a == b
