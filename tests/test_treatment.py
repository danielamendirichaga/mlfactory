"""Tests for the v2 A/B-test simulation (S14) — randomization, heterogeneous uplift, oracle cols.

The generator overlays a randomized offer with a *heterogeneous* effect τ(x); we verify the
experiment is clean, the four uplift quadrants exist (incl. sleeping dogs), the observed churn is
the factual outcome, and the ground-truth columns can never leak into features. v1 is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.model import feature_columns


@pytest.fixture(scope="module")
def panel_t():
    return make_panel(n_accounts=2500, n_months=12, seed=5, treatment=True)


# --- v1 is untouched when treatment is off ------------------------------- #
def test_default_frame_unchanged():
    a = make_panel(n_accounts=600, n_months=8, seed=7)
    b = make_panel(n_accounts=600, n_months=8, seed=7, treatment=False)
    pd.testing.assert_frame_equal(a, b)
    assert "treated" not in a.columns and "true_uplift" not in a.columns
    assert 0.05 < a["churn_next_30d"].mean() < 0.16  # still ~10%


def test_treatment_adds_columns_and_is_deterministic(panel_t):
    for col in ("treated", "true_uplift", "churn_if_control", "churn_if_treated"):
        assert col in panel_t.columns
    again = make_panel(n_accounts=2500, n_months=12, seed=5, treatment=True)
    pd.testing.assert_frame_equal(panel_t, again)


# --- the experiment is clean --------------------------------------------- #
def test_randomization_is_balanced_and_feature_independent(panel_t):
    assert 0.46 < panel_t["treated"].mean() < 0.54
    # treatment assignment is independent of features → a valid experiment
    corr = panel_t["treated"].corr(panel_t["product_usage_hours_30d"].fillna(0))
    assert abs(corr) < 0.06


# --- heterogeneous effect with all four quadrants ------------------------ #
def test_uplift_is_heterogeneous_with_sleeping_dogs(panel_t):
    tu = panel_t["true_uplift"]
    assert tu.std() > 0.02  # genuinely heterogeneous, not a constant lift
    assert (tu > 0.05).any()  # persuadables (offer strongly helps)
    assert (tu < -0.01).any()  # sleeping dogs (offer backfires → negative uplift)


def test_uplift_is_not_determined_by_risk(panel_t):
    # Among would-be churners (same "high risk"), uplift still spans both signs →
    # uplift ranks customers differently than risk does. That is the whole point.
    churners = panel_t[panel_t["churn_if_control"] == 1]["true_uplift"]
    assert (churners > 0.03).any() and (churners < -0.005).any()
    assert churners.std() > 0.02


# --- outcomes are consistent --------------------------------------------- #
def test_observed_churn_is_the_factual_arm(panel_t):
    factual = np.where(
        panel_t["treated"] == 1, panel_t["churn_if_treated"], panel_t["churn_if_control"]
    )
    assert (panel_t["churn_next_30d"].to_numpy() == factual).all()


def test_ate_matches_true_uplift_and_is_positive(panel_t):
    ate = panel_t["churn_if_control"].mean() - panel_t["churn_if_treated"].mean()
    assert ate == pytest.approx(panel_t["true_uplift"].mean(), abs=0.015)
    assert ate > 0  # on net, offers reduce churn (persuadables outweigh sleeping dogs)


# --- oracle columns can never become features (config-driven, generic core) ---- #
def test_oracle_and_treatment_cols_excluded_via_config(panel_t):
    """The generic core drops only what the config's `exclude_columns` declares — the SaaS
    domain lists its oracle + treatment ground-truth columns there, so the core hardcodes no
    domain column names."""
    oracle = ["treated", "true_uplift", "churn_if_control", "churn_if_treated"]
    cfg = ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {
                "id_col": "account_id",
                "target_col": "churn_next_30d",
                "date_col": "observation_month",
                "value_col": "cltv",
                "features": "auto",
                "exclude_columns": oracle,
            },
        }
    )
    numeric, categorical = feature_columns(panel_t, cfg)
    feats = set(numeric) | set(categorical)
    for leak in oracle:
        assert leak not in feats
    # Without the declaration the generic core has no domain knowledge to drop them —
    # proving the exclusion is config-driven, not hardcoded.
    bare = cfg.model_copy(deep=True)
    bare.columns.exclude_columns = []
    feats_bare = set(feature_columns(panel_t, bare)[0]) | set(feature_columns(panel_t, bare)[1])
    assert "treated" in feats_bare
