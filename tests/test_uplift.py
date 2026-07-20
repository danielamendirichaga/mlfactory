"""Tests for uplift meta-learners (S15) — recovery of the planted τ, S vs T, guards, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel
from mlfactory.domains.saas.uplift import (
    UpliftCard,
    UpliftError,
    load_uplift,
    save_uplift,
    train_uplift,
)

FEATURES = [
    "tenure_months",
    "mrr",
    "product_usage_hours_30d",
    "days_since_last_login",
    "usage_trend_30d",
    "support_tickets_30d",
    "payment_failures_30d",
    "discount_months_left",
    "in_discount",
    "plan_tier",
    "region",
]


def _cfg():
    return ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {
                "id_col": "account_id",
                "target_col": "churn_next_30d",
                "date_col": "observation_month",
                "value_col": "cltv",
                "features": FEATURES,
            },
        }
    )


@pytest.fixture(scope="module")
def fitted():
    df = make_panel(n_accounts=3500, n_months=12, seed=9, treatment=True)
    cfg = _cfg()
    t_model, t_card = train_uplift(df, cfg, learner="t", seed=1)
    s_model, s_card = train_uplift(df, cfg, learner="s", seed=1)
    return df, cfg, t_model, t_card, s_model, s_card


def test_t_learner_recovers_true_uplift(fitted):
    df, _, _, t_card, *_ = fitted
    assert t_card.tau_recovery_corr is not None and t_card.tau_recovery_corr > 0.25
    # ATE is easy; the learner should match the true average effect closely.
    assert t_card.ate_hat == pytest.approx(float(df["true_uplift"].mean()), abs=0.012)
    assert t_card.ate_hat > 0


def test_ranks_persuadables_above_sleeping_dogs(fitted):
    df, _, t_model, *_ = fitted
    tau = t_model.predict_uplift(df)
    tu = df["true_uplift"].to_numpy()
    assert tau[tu > 0.03].mean() > tau[tu < -0.005].mean()  # persuadables score higher


def test_t_learner_beats_s_learner_on_heterogeneity(fitted):
    _, _, _, t_card, _, s_card = fitted
    # Both recover the ATE, but the T-learner captures the *variation* far better.
    assert t_card.tau_recovery_corr > s_card.tau_recovery_corr


def test_requires_treatment_column():
    plain = make_panel(n_accounts=400, n_months=6, seed=2)  # no --treatment
    with pytest.raises(UpliftError, match="treated"):
        train_uplift(plain, _cfg())


def test_unknown_learner_raises(fitted):
    df, cfg, *_ = fitted
    with pytest.raises(UpliftError, match="learner"):
        train_uplift(df, cfg, learner="x")


def test_card_artifact_and_features(fitted):
    _, _, _, t_card, *_ = fitted
    assert t_card.artifact == "uplift-card"
    assert t_card.parent_sha256 and len(t_card.parent_sha256) == 64
    assert t_card.n_treated + t_card.n_control == t_card.n_train
    for leak in ("treated", "true_uplift", "churn_if_control", "churn_if_treated"):
        assert leak not in t_card.features


def test_card_roundtrip_and_model_persistence(fitted, tmp_path):
    df, _, t_model, t_card, *_ = fitted
    p = tmp_path / "uplift.pkl"
    save_uplift(t_model, p)
    reloaded = load_uplift(p)
    assert np.allclose(reloaded.predict_uplift(df), t_model.predict_uplift(df))
    jp = tmp_path / "uplift.card.json"
    t_card.write_json(jp)
    assert UpliftCard.model_validate_json(jp.read_text()) == t_card
