"""Tests for the copilot's recommendation rules (S19) — deterministic judgment-support."""

from __future__ import annotations

import pandas as pd

from mlfactory.config import ChurnConfig
from mlfactory.recommend import (
    recommend_experiment,
    recommend_features,
    recommend_model,
    recommend_policy,
    recommend_retrain,
    recommend_ship,
    recommend_split,
)


def _cfg(**schema_extra) -> ChurnConfig:
    schema = {"id_col": "id", "target_col": "y"}
    schema.update(schema_extra)
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


# --- features / leakage --------------------------------------------------- #
def test_features_flags_the_leak():
    records = [{"column": "leak", "target_corr": 0.92}, {"column": "genuine", "target_corr": 0.2}]
    rec = recommend_features(records)
    assert rec.gate == "features" and rec.action["exclude"] == ["leak"]
    assert "leak" in rec.recommendation


def test_features_keeps_clean_set():
    records = [{"column": "a", "target_corr": 0.1}, {"column": "b", "target_corr": -0.3}]
    rec = recommend_features(records)
    assert rec.action["exclude"] == [] and "Keep all" in rec.recommendation


# --- split ---------------------------------------------------------------- #
def test_split_time_when_dated_random_when_snapshot():
    assert recommend_split(_cfg(date_col="d")).action["strategy"] == "time"
    assert recommend_split(_cfg()).action["strategy"] == "random"


# --- model (stability over peak AUC) ------------------------------------- #
def test_model_prefers_stable_over_overfit():
    rows = [
        {"model": "logistic", "holdout_auc": 0.675, "auc_drop": 0.012, "stable": True},
        {"model": "xgboost", "holdout_auc": 0.690, "auc_drop": 0.160, "stable": False},
    ]
    rec = recommend_model(rows)
    assert rec.action["model"] == "logistic"  # stable wins despite lower AUC
    assert "xgboost" in rec.rationale and "overfit" in rec.rationale


def test_model_picks_best_auc_among_stable():
    rows = [
        {"model": "a", "holdout_auc": 0.70, "auc_drop": 0.01, "stable": True},
        {"model": "b", "holdout_auc": 0.72, "auc_drop": 0.02, "stable": True},
    ]
    assert recommend_model(rows).action["model"] == "b"


def test_model_handles_empty():
    assert recommend_model([]).action["model"] is None


# --- policy --------------------------------------------------------------- #
def test_policy_needs_value_col():
    assert recommend_policy(_cfg(value_col="cltv")).action["value_col"] == "cltv"
    assert recommend_policy(_cfg()).action["value_col"] is None


# --- retrain (from drift) ------------------------------------------------- #
def test_retrain_on_drift():
    drift = {"skipped": False, "retrain_recommended": True, "drifted": ["watch_hours_30d", "x"]}
    rec = recommend_retrain(drift)
    assert rec.action["retrain"] is True and "2 feature" in rec.rationale


def test_no_retrain_without_drift():
    assert (
        recommend_retrain({"skipped": False, "retrain_recommended": False, "drifted": []}).action[
            "retrain"
        ]
        is False
    )


def test_retrain_unavailable_on_snapshot():
    rec = recommend_retrain({"skipped": True})
    assert rec.action["retrain"] is False and "unavailable" in rec.recommendation.lower()


# --- ship (go / no-go) ---------------------------------------------------- #
def test_ship_go_when_metrics_clear_the_bar():
    assert recommend_ship({"metrics": {"auc": 0.83, "ece": 0.03}}).action["ship"] is True


def test_ship_nogo_on_low_auc_or_bad_calibration():
    assert recommend_ship({"metrics": {"auc": 0.55, "ece": 0.03}}).action["ship"] is False
    assert recommend_ship({"metrics": {"auc": 0.83, "ece": 0.20}}).action["ship"] is False


# --- experiment / v1-vs-v2 ------------------------------------------------ #
def test_experiment_available_with_a_randomized_treatment():
    df = pd.DataFrame({"treated": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    rec = recommend_experiment(df)
    assert rec.action["uplift"] is True and "available" in rec.recommendation.lower()


def test_experiment_absent_means_v1_only():
    df = pd.DataFrame({"x": [1, 2, 3]})  # no treated column → observational
    rec = recommend_experiment(df)
    assert rec.action["uplift"] is False and "v1" in rec.recommendation.lower()
