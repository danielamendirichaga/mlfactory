"""Tests for the risk-vs-uplift policy contrast (S17) — the v2 payoff, scored on ground truth."""

from __future__ import annotations

import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.model import train_model
from mlfactory.policy import PolicyContrast, PolicyError, contrast_policies
from mlfactory.uplift import train_uplift

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


def _cfg(value_col: str | None = "cltv"):
    schema = {
        "id_col": "account_id",
        "target_col": "churn_next_30d",
        "date_col": "observation_month",
        "features": FEATURES,
    }
    if value_col is not None:
        schema["value_col"] = value_col
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


@pytest.fixture(scope="module")
def models():
    df = make_panel(n_accounts=4000, n_months=12, seed=9, treatment=True)
    cfg = _cfg()
    risk, _ = train_model(df, cfg, model="logistic", seed=1)
    up, _ = train_uplift(df, cfg, learner="t", seed=1)
    return df, cfg, risk, up


def test_uplift_beats_risk_on_true_net_value(models):
    df, cfg, risk, up = models
    c = contrast_policies(risk, up, df, cfg, offer_cost=3.0, n_offers=1500)
    r, u = c.strategies["risk"], c.strategies["uplift"]
    # equal budget → equal offers made
    assert r["n_targeted"] == u["n_targeted"] == 1500
    # targeting persuadables retains more *true* value than targeting the highest-risk
    assert u["true_net_value"] > r["true_net_value"]
    assert c.uplift_net_advantage == round(u["true_net_value"] - r["true_net_value"], 2)
    assert c.uplift_net_advantage > 0


def test_uplift_treats_fewer_sleeping_dogs(models):
    df, cfg, risk, up = models
    c = contrast_policies(risk, up, df, cfg, offer_cost=3.0, n_offers=1500)
    assert (
        c.strategies["uplift"]["sleeping_dogs_treated"]
        < c.strategies["risk"]["sleeping_dogs_treated"]
    )
    assert c.sleeping_dogs_avoided > 0


def test_requires_true_uplift(models):
    _, cfg, risk, up = models
    plain = make_panel(n_accounts=300, n_months=6, seed=2)  # no A/B columns
    with pytest.raises(PolicyError, match="true_uplift"):
        contrast_policies(risk, up, plain, cfg, n_offers=50)


def test_requires_value_col(models):
    df, _, risk, up = models
    with pytest.raises(PolicyError, match="value_col"):
        contrast_policies(risk, up, df, _cfg(value_col=None), n_offers=50)


def test_artifact_lineage_and_roundtrip(models, tmp_path):
    df, cfg, risk, up = models
    c = contrast_policies(risk, up, df, cfg, offer_cost=3.0, budget=6000.0)
    assert c.artifact == "policy-contrast"
    assert c.parent_sha256 and len(c.parent_sha256) == 64
    p = tmp_path / "policy-contrast.json"
    c.write_json(p)
    assert PolicyContrast.model_validate_json(p.read_text()) == c
