"""Tests for the retention policy simulator (S10) — cost-based, budget-constrained targeting."""

from __future__ import annotations

import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.model import train_model
from mlfactory.policy import PolicyError, PolicyReport, simulate_policy

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


def _cfg(value_col: str | None = "cltv"):
    schema = dict(SCHEMA)
    if value_col is None:
        schema.pop("value_col")
    else:
        schema["value_col"] = value_col
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


@pytest.fixture(scope="module")
def fitted():
    train = make_panel(n_accounts=1500, n_months=10, seed=41)
    data = make_panel(n_accounts=800, n_months=10, seed=42)
    est, _ = train_model(train, _cfg(), model="logistic", seed=1)
    return est, data


def test_unlimited_targets_all_profitable(fitted):
    est, data = fitted
    r = simulate_policy(est, data, _cfg(), save_rate=0.3, offer_cost=2.0)
    assert r.n_targeted == r.n_eligible
    assert r.n_eligible > 0  # some customers are worth an offer
    # Net = retained − spend, and every targeted customer is individually profitable → net > 0.
    assert r.net_value == round(r.expected_retained_value - r.expected_spend, 2)
    assert r.net_value > 0


def test_n_offers_caps_the_target(fitted):
    est, data = fitted
    r = simulate_policy(est, data, _cfg(), offer_cost=2.0, n_offers=50)
    assert r.n_targeted == min(50, r.n_eligible)


def test_budget_caps_the_spend(fitted):
    est, data = fitted
    r = simulate_policy(est, data, _cfg(), offer_cost=5.0, budget=100.0)
    assert r.n_targeted <= 100 // 5  # 20 offers max
    assert r.expected_spend <= 100.0


def test_requires_cltv_column(fitted):
    est, data = fitted
    with pytest.raises(PolicyError, match="value_col"):
        simulate_policy(est, data, _cfg(value_col=None))


def test_budget_and_n_offers_conflict(fitted):
    est, data = fitted
    with pytest.raises(PolicyError, match="not both"):
        simulate_policy(est, data, _cfg(), budget=100.0, n_offers=10)


def test_segments_sum_to_targeted(fitted):
    est, data = fitted
    r = simulate_policy(est, data, _cfg(), offer_cost=2.0, n_offers=100)
    assert sum(s["n_targeted"] for s in r.segments.values()) == r.n_targeted


def test_artifact_lineage_and_roundtrip(fitted, tmp_path):
    est, data = fitted
    r = simulate_policy(est, data, _cfg(), offer_cost=2.0)
    assert r.artifact == "policy-report"
    assert r.parent_sha256 and len(r.parent_sha256) == 64
    p = tmp_path / "policy-report.json"
    r.write_json(p)
    assert PolicyReport.model_validate_json(p.read_text()) == r


def test_higher_offer_cost_shrinks_eligible(fitted):
    est, data = fitted
    cheap = simulate_policy(est, data, _cfg(), offer_cost=1.0)
    dear = simulate_policy(est, data, _cfg(), offer_cost=20.0)
    assert dear.n_eligible <= cheap.n_eligible
