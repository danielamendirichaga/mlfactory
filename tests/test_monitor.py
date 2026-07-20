"""Tests for drift monitoring (S12) — per-feature PSI + retrain flag + graceful snapshot skip."""

from __future__ import annotations

import pytest

from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import make_panel
from mlfactory.domains.saas.monitor import DriftReport, monitor_drift

PANEL_SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
}


def _cfg(schema=PANEL_SCHEMA):
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


@pytest.fixture(scope="module")
def panel():
    # 18 monthly cohorts → the engineered watch-hours drift is pronounced.
    return make_panel(n_accounts=2000, n_months=18, seed=61)


def test_flags_the_drifting_feature(panel):
    report = monitor_drift(panel, _cfg(), threshold=0.25)
    assert report.mode == "panel" and not report.skipped
    assert report.retrain_recommended is True
    # product_usage_hours_30d drifts on purpose → should be flagged.
    assert "product_usage_hours_30d" in report.drifted
    watch = next(r for r in report.features if r["feature"] == "product_usage_hours_30d")
    assert watch["psi"] > 0.25 and watch["status"] == "major"


def test_stable_feature_not_flagged(panel):
    report = monitor_drift(panel, _cfg())
    age = next(r for r in report.features if r["feature"] == "company_size_employees")
    assert age["psi"] < 0.1  # age is not engineered to drift
    assert "company_size_employees" not in report.drifted


def test_features_sorted_by_psi_desc(panel):
    report = monitor_drift(panel, _cfg())
    psis = [r["psi"] for r in report.features]
    assert psis == sorted(psis, reverse=True)


def test_threshold_controls_retrain(panel):
    low = monitor_drift(panel, _cfg(), threshold=0.1)
    high = monitor_drift(panel, _cfg(), threshold=0.9)
    assert len(low.drifted) >= len(high.drifted)


def test_snapshot_mode_skips_gracefully(panel):
    schema = {k: v for k, v in PANEL_SCHEMA.items() if k != "date_col"}
    report = monitor_drift(panel, _cfg(schema))
    assert report.skipped is True
    assert report.mode == "snapshot"
    assert report.retrain_recommended is False
    assert report.features == []


def test_artifact_lineage_and_roundtrip(panel, tmp_path):
    report = monitor_drift(panel, _cfg())
    assert report.artifact == "drift-report"
    assert report.parent_sha256 and len(report.parent_sha256) == 64
    p = tmp_path / "drift-report.json"
    report.write_json(p)
    assert DriftReport.model_validate_json(p.read_text()) == report
