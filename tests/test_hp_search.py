"""Tests for Optuna hp-search + the hist_gbm engine (#4)."""

from __future__ import annotations

import pytest

from mlfactory.compute.hp_search import optuna_search
from mlfactory.compute.model import ModelError, train_model
from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel

FEATURES = [
    "tenure_months",
    "mrr",
    "product_usage_hours_30d",
    "days_since_last_login",
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
def train_df():
    return make_panel(n_accounts=600, n_months=8, seed=11)


def test_optuna_search_is_seeded_reproducible(train_df):
    _, params_a, score_a = optuna_search(train_df, _cfg(), "logistic", n_trials=5, seed=1)
    _, params_b, score_b = optuna_search(train_df, _cfg(), "logistic", n_trials=5, seed=1)
    assert params_a == params_b  # same seed → same winner
    assert score_a == pytest.approx(score_b)


def test_optuna_search_beats_floor(train_df):
    est, _, score = optuna_search(train_df, _cfg(), "logistic", n_trials=5, seed=1)
    assert 0.5 < score <= 1.0
    assert est.predict_proba(train_df[FEATURES])[:, 1].shape[0] == len(train_df)


def test_train_optuna_path(train_df):
    _, card = train_model(train_df, _cfg(), model="logistic", optuna=True, n_trials=5, seed=1)
    assert card.tuned is True and "C" in card.hyperparams
    assert card.train_metrics["auc"] > card.baseline_metrics["auc"]


def test_optuna_rejects_tree(train_df):
    with pytest.raises(ModelError, match="Optuna"):
        train_model(train_df, _cfg(), model="tree", optuna=True)


def test_optuna_mutual_exclusion(train_df):
    with pytest.raises(ModelError, match="cannot be combined"):
        train_model(train_df, _cfg(), model="xgboost", optuna=True, tune=True)


def test_hist_gbm_fits_and_beats_floor(train_df):
    _, card = train_model(train_df, _cfg(), model="hist_gbm", seed=1)
    assert card.model_family == "hist_gbm"
    assert card.train_metrics["auc"] > card.baseline_metrics["auc"]
