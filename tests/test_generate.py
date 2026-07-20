"""Tests for the synthetic SaaS churn panel generator (S2).

Small sizes keep it fast; the DGP behaviour (determinism + the 4 levers) is what matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfactory.generate import DRIFT_FEATURE, TARGET, make_panel

EXPECTED_COLUMNS = {
    "account_id",
    "observation_month",
    "signup_month",
    "tenure_months",
    "plan_tier",
    "mrr",
    "payment_method",
    "in_discount",
    "discount_months_left",
    "seats_licensed",
    "product_usage_hours_30d",
    "active_days_30d",
    "days_since_last_login",
    "usage_trend_30d",
    "key_actions_30d",
    "features_adopted_30d",
    "avg_session_minutes",
    "support_tickets_30d",
    "payment_failures_30d",
    "company_size_employees",
    "region",
    "acquisition_channel",
    "cancel_page_visits_30d",
    "cltv",
    "churn_next_30d",
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(n_accounts=1500, n_months=18, seed=7)


def test_deterministic():
    a = make_panel(n_accounts=800, n_months=12, seed=1)
    b = make_panel(n_accounts=800, n_months=12, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = make_panel(n_accounts=800, n_months=12, seed=1)
    b = make_panel(n_accounts=800, n_months=12, seed=2)
    assert not a.equals(b)


def test_schema(panel):
    assert set(panel.columns) == EXPECTED_COLUMNS
    assert set(panel[TARGET].unique()) <= {0, 1}
    assert pd.api.types.is_datetime64_any_dtype(panel["observation_month"])
    # Panel key is unique per account-month; ids repeat across months.
    assert not panel.duplicated(["account_id", "observation_month"]).any()
    assert panel.groupby("account_id").size().max() > 1  # genuinely panel


def test_churn_rate_imbalanced(panel):
    rate = panel[TARGET].mean()
    assert 0.07 <= rate <= 0.13, f"churn rate {rate:.3f} outside target band"


def test_drift_usage_declines(panel):
    by_cohort = panel.groupby("observation_month")[DRIFT_FEATURE].mean()
    idx = np.arange(len(by_cohort))
    # A clear downward trend across cohorts (robust), plus a meaningful magnitude.
    # (The PSI-based drift check lands in S5 once the metric core exists.)
    corr = np.corrcoef(idx, by_cohort.to_numpy())[0, 1]
    assert corr < -0.7
    assert by_cohort.iloc[-1] < 0.85 * by_cohort.iloc[0]


def test_missingness(panel):
    age_null = panel["company_size_employees"].isna().mean()
    assert 0.04 <= age_null <= 0.12
    assert panel["product_usage_hours_30d"].isna().mean() > 0  # blank for some new subs


def test_leakage_trap_separates_churners(panel):
    churner = panel.loc[panel[TARGET] == 1, "cancel_page_visits_30d"].mean()
    stayer = panel.loc[panel[TARGET] == 0, "cancel_page_visits_30d"].mean()
    assert churner > 1.0
    assert stayer < 0.2
    assert churner > 10 * stayer  # near-perfectly predictive (the planted trap)


def test_churn_is_last_row_per_account(panel):
    """A churned account's churn_next_30d=1 row must be their LAST (truncation correct)."""
    churn_months = panel.loc[panel[TARGET] == 1].set_index("account_id")["observation_month"]
    last_months = panel.groupby("account_id")["observation_month"].max()
    aligned = last_months.loc[churn_months.index]
    assert (churn_months.values == aligned.values).all()
    # And a account churns at most once.
    assert panel.loc[panel[TARGET] == 1].groupby("account_id").size().max() == 1


def test_tenure_and_values_sane(panel):
    assert (panel["tenure_months"] >= 1).all()
    assert (panel["cltv"] > 0).all()
    assert (panel["discount_months_left"] >= 0).all()
    assert (panel["usage_trend_30d"].abs() < 500).all()
    assert np.isfinite(panel["cltv"]).all()
