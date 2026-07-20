"""Tests for per-column profiling / EDA (S4)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel
from mlfactory.compute.profile import high_corr_features, infer_role, profile_frame

SYNTH_SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    df = make_panel(n_accounts=800, n_months=12, seed=3)
    cfg = ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SYNTH_SCHEMA})
    return profile_frame(df, cfg)


def _rec(records, col):
    return next(r for r in records if r["column"] == col)


def test_roles_from_config_and_dtype(records):
    assert _rec(records, "account_id")["role"] == "id"  # panel id, not "numeric"
    assert _rec(records, "observation_month")["role"] == "datetime"
    assert _rec(records, "churn_next_30d")["role"] == "target"
    assert _rec(records, "plan_tier")["role"] == "categorical"
    assert _rec(records, "product_usage_hours_30d")["role"] == "numeric"


def test_infer_role_standalone():
    assert infer_role(pd.Series([1.0, 2.0])) == "numeric"
    assert infer_role(pd.Series(["a", "b"])) == "categorical"
    assert infer_role(pd.Series([True, False])) == "categorical"
    assert infer_role(pd.to_datetime(pd.Series(["2023-01-01"]))) == "datetime"


def test_numeric_stats_present(records):
    w = _rec(records, "product_usage_hours_30d")
    assert {"min", "max", "mean", "std", "q25", "q50", "q75"} <= set(w)
    assert w["min"] <= w["q50"] <= w["max"]


def test_null_rate_reported(records):
    assert _rec(records, "company_size_employees")["null_rate"] > 0
    assert _rec(records, "plan_tier")["null_rate"] == 0.0


def test_target_corr_signs(records):
    # More watching → less churn (negative); more cancel-page visits → more churn (positive).
    assert _rec(records, "product_usage_hours_30d")["target_corr"] < 0
    assert _rec(records, "days_since_last_login")["target_corr"] > 0
    assert _rec(records, "cancel_page_visits_30d")["target_corr"] > 0.3  # the planted trap
    # The target itself carries no self-correlation.
    assert "target_corr" not in _rec(records, "churn_next_30d")


def test_id_and_datetime_have_no_target_corr(records):
    assert "target_corr" not in _rec(records, "account_id")
    assert "target_corr" not in _rec(records, "observation_month")


def test_high_corr_flags_the_leak(records):
    leaky = dict(high_corr_features(records, threshold=0.5))
    assert "cancel_page_visits_30d" in leaky
    # Genuine drivers stay below the leakage threshold.
    assert "product_usage_hours_30d" not in leaky


def test_target_corr_works_with_string_labels():
    # Real churn data often uses "Yes"/"No" labels (e.g. IBM Telco), not 0/1. The correlation
    # must binarize via positive_value instead of coercing the string to a number (all-NaN).
    df = pd.DataFrame(
        {
            "cust": range(10),
            "tenure": list(range(1, 11)),
            "Churn": ["Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No"],
        }
    )
    cfg = ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {"id_col": "cust", "target_col": "Churn", "positive_value": "Yes"},
        }
    )
    tenure = next(r for r in profile_frame(df, cfg) if r["column"] == "tenure")
    assert tenure["target_corr"] is not None  # a string target no longer blanks it out
    assert tenure["target_corr"] < 0  # low tenure → churned (Yes) → negative correlation
