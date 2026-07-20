"""Tests for the model layer (S7): the menu, leakage-safe pipeline, baseline floor, model-card."""

from __future__ import annotations

import numpy as np
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.model import (
    MODELS,
    ModelCard,
    ModelError,
    load_model,
    train_model,
)

# Explicit features EXCLUDING the leakage trap → tests exercise genuine signal.
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
def train_df():
    return make_panel(n_accounts=1500, n_months=10, seed=11)


def _cfg():
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SCHEMA})


@pytest.mark.parametrize("model", MODELS)
def test_each_model_fits_and_beats_floor(train_df, model):
    est, card = train_model(train_df, _cfg(), model=model)
    proba = est.predict_proba(train_df[FEATURES])[:, 1]
    assert proba.shape[0] == len(train_df)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert card.train_metrics["auc"] > card.baseline_metrics["auc"]  # beats the floor
    assert card.model_family == model
    assert card.n_features == len(FEATURES)


def test_baseline_floor_is_chance(train_df):
    _, card = train_model(train_df, _cfg(), model="logistic")
    assert abs(card.baseline_metrics["auc"] - 0.5) < 1e-6


def test_smote_path(train_df):
    _, card = train_model(train_df, _cfg(), model="logistic", smote=True)
    assert card.smote is True


def test_calibrate_path(train_df):
    _, card = train_model(train_df, _cfg(), model="rf", calibrate=True)
    assert card.calibrated is True


def test_smote_rejected_for_tree(train_df):
    with pytest.raises(ModelError, match="tree"):
        train_model(train_df, _cfg(), model="tree", smote=True)


def test_unknown_model_raises(train_df):
    with pytest.raises(ModelError, match="unknown model"):
        train_model(train_df, _cfg(), model="neural_net")


def test_model_card_artifact_and_lineage(train_df, tmp_path):
    _, card = train_model(train_df, _cfg(), model="logistic")
    assert card.artifact == "model-card"
    assert card.parent_sha256 and len(card.parent_sha256) == 64
    assert set(card.features) == set(FEATURES)
    p = tmp_path / "model.card.json"
    card.write_json(p)
    assert ModelCard.model_validate_json(p.read_text()) == card


def test_save_load_roundtrip(train_df, tmp_path):
    est, _ = train_model(train_df, _cfg(), model="rf")
    p = tmp_path / "model.pkl"
    from mlfactory.model import save_model

    save_model(est, p)
    reloaded = load_model(p)
    a = est.predict_proba(train_df[FEATURES])[:, 1]
    b = reloaded.predict_proba(train_df[FEATURES])[:, 1]
    assert np.allclose(a, b)


def test_tune_paths_small(train_df):
    # Exercise the standard search on tree (ccp) + xgboost (grid) on small data.
    _, tree_card = train_model(train_df, _cfg(), model="tree", tune=True)
    assert "ccp_alpha" in tree_card.hyperparams
    _, xgb_card = train_model(train_df, _cfg(), model="xgboost", tune=True)
    assert {"learning_rate", "max_depth", "n_estimators"} <= set(xgb_card.hyperparams)


def test_early_stopping_panel_uses_time_inner_split(train_df):
    est, card = train_model(train_df, _cfg(), model="xgboost", early_stopping=True)
    assert card.early_stopping is True
    assert card.hyperparams["inner_val"] == "time"  # panel → time-aware inner-val
    assert 1 <= card.hyperparams["best_iteration"] <= card.hyperparams["n_estimators_max"]
    assert est.predict_proba(train_df[FEATURES])[:, 1].shape[0] == len(train_df)


def test_early_stopping_snapshot_uses_stratified(train_df):
    # A config without date_col → snapshot mode → stratified inner-val (the standard method).
    snap = {k: v for k, v in SCHEMA.items() if k != "date_col"}
    cfg = ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": snap})
    _, card = train_model(train_df, cfg, model="xgboost", early_stopping=True)
    assert card.hyperparams["inner_val"] == "stratified"


def test_early_stopping_only_xgboost(train_df):
    with pytest.raises(ModelError, match="only supported for the xgboost"):
        train_model(train_df, _cfg(), model="rf", early_stopping=True)


def test_early_stopping_excludes_tune(train_df):
    with pytest.raises(ModelError, match="cannot be combined"):
        train_model(train_df, _cfg(), model="xgboost", early_stopping=True, tune=True)


def test_cli_train_warns_on_leakage(tmp_path, train_df):
    from typer.testing import CliRunner

    from mlfactory.cli import app

    p = tmp_path / "train.parquet"
    train_df.to_parquet(p, index=False)
    cfgp = tmp_path / "churn.yaml"
    cfgp.write_text(
        "source:\n  kind: synthetic\n"
        "schema:\n  id_col: account_id\n  target_col: churn_next_30d\n"
        "  date_col: observation_month\n  features: auto\n"
    )
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--train",
            str(p),
            "--config",
            str(cfgp),
            "--model",
            "logistic",
            "--model-out",
            str(tmp_path / "m.pkl"),
        ],
    )
    assert result.exit_code == 0
    assert "leakage" in result.output
    assert "cancel_page_visits_30d" in result.output
