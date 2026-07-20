"""Tests for the data validator (S3): does the dataset match the config and is it usable?"""

from __future__ import annotations

import pandas as pd

from mlfactory.config import ChurnConfig
from mlfactory.validate import validate


def _cfg(schema: dict) -> ChurnConfig:
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


PANEL = pd.DataFrame(
    {
        "account_id": [1, 1, 2],
        "observation_month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-01-01"]),
        "churn_next_30d": [0, 1, 0],
        "product_usage_hours_30d": [5.0, 2.0, 8.0],
    }
)
PANEL_SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
}


def test_clean_panel_is_usable():
    report = validate(PANEL, _cfg(PANEL_SCHEMA))
    assert report.ok
    assert report.mode == "panel"
    target = next(c for c in report.checks if c.name == "target")
    assert target.status == "pass"


def test_experiment_check_guides_v1_vs_v2():
    # observational data (no treated column) → v1 (risk) pipeline
    v1 = next(c for c in validate(PANEL, _cfg(PANEL_SCHEMA)).checks if c.name == "experiment")
    assert "v1 (risk)" in v1.message
    # a randomized `treated` column → uplift (v2) available
    ab = PANEL.assign(treated=[0, 1, 0])
    v2 = next(c for c in validate(ab, _cfg(PANEL_SCHEMA)).checks if c.name == "experiment")
    assert "uplift (v2) available" in v2.message


def test_missing_target_fails():
    report = validate(PANEL, _cfg({"id_col": "account_id", "target_col": "cancelled"}))
    assert not report.ok
    assert any(c.name == "target" and c.status == "fail" for c in report.checks)


def test_single_class_target_fails():
    df = PANEL.assign(churn_next_30d=[0, 0, 0])
    report = validate(df, _cfg(PANEL_SCHEMA))
    assert not report.ok


def test_positive_value_absent_fails():
    df = pd.DataFrame({"id": [1, 2], "churn": ["no", "yes"]})
    report = validate(df, _cfg({"id_col": "id", "target_col": "churn", "positive_value": 1}))
    assert not report.ok  # positive_value 1 not among {"no","yes"}


def test_string_positive_value_passes():
    df = pd.DataFrame({"id": [1, 2, 3], "churn": ["no", "yes", "no"]})
    cfg = _cfg({"id_col": "id", "target_col": "churn", "positive_value": "yes"})
    report = validate(df, cfg)
    assert report.ok
    assert report.mode == "snapshot"


def test_declared_date_col_absent_fails():
    schema = {"id_col": "account_id", "target_col": "churn_next_30d", "date_col": "missing_col"}
    report = validate(PANEL, _cfg(schema))
    assert not report.ok
    assert any(c.name == "date" and c.status == "fail" for c in report.checks)


def test_snapshot_mode_note():
    df = pd.DataFrame({"id": [1, 2, 3], "churn": [0, 1, 0]})
    report = validate(df, _cfg({"id_col": "id", "target_col": "churn"}))
    assert report.ok
    assert report.mode == "snapshot"
    date = next(c for c in report.checks if c.name == "date")
    assert "snapshot" in date.message


def test_duplicate_snapshot_ids_warn():
    df = pd.DataFrame({"id": [1, 1, 2], "churn": [0, 1, 0]})
    report = validate(df, _cfg({"id_col": "id", "target_col": "churn"}))
    assert report.ok  # a warning, not a failure
    idc = next(c for c in report.checks if c.name == "id")
    assert idc.status == "warn"


def test_empty_dataset_fails():
    df = pd.DataFrame({"account_id": [], "churn_next_30d": []})
    report = validate(df, _cfg({"id_col": "account_id", "target_col": "churn_next_30d"}))
    assert not report.ok


def test_render_has_symbols():
    text = validate(PANEL, _cfg(PANEL_SCHEMA)).render()
    assert "✔" in text
    assert "USABLE" in text
