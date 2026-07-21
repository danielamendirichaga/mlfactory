"""Slice S1 — the leakage-drop actually reaches the pipeline.

`set_exclude_columns` writes a confirmed EDA drop into `schema.exclude_columns` in churn.yaml
(comment-preserving), and `feature_columns` then drops it. This closes the gap where the drop
lived only in the eda-exploration artifact and `train` silently kept training on the leak.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.compute.model import feature_columns
from mlfactory.config import CONFIG_TEMPLATE, ConfigError, load_config, set_exclude_columns


def _template(tmp_path: Path) -> Path:
    p = tmp_path / "churn.yaml"
    p.write_text(CONFIG_TEMPLATE)
    return p


def test_add_from_commented_default(tmp_path: Path) -> None:
    p = _template(tmp_path)
    new = set_exclude_columns(p, add=["cancel_page_visits_30d"])
    assert new == ["cancel_page_visits_30d"]
    assert load_config(p).columns.exclude_columns == ["cancel_page_visits_30d"]
    # onboarding comments survive the targeted edit
    assert "# mlfactory config" in p.read_text()


def test_add_is_idempotent_and_ordered(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_exclude_columns(p, add=["a"])
    assert set_exclude_columns(p, add=["a", "b"]) == ["a", "b"]


def test_remove(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_exclude_columns(p, add=["a", "b"])
    assert set_exclude_columns(p, remove=["a"]) == ["b"]


def test_set_replaces_the_whole_list(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_exclude_columns(p, add=["a", "b"])
    assert set_exclude_columns(p, replace=["c"]) == ["c"]


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        set_exclude_columns(tmp_path / "nope.yaml", add=["x"])


def test_drop_propagates_to_feature_columns(tmp_path: Path) -> None:
    """The point of the slice: after recording the drop, the model never sees the column."""
    p = _template(tmp_path)
    set_exclude_columns(p, add=["cancel_page_visits_30d"])
    cfg = load_config(p)
    df = pd.DataFrame(
        {
            "account_id": [1, 2],
            "observation_month": ["2021-01", "2021-02"],
            "churn_next_30d": [0, 1],
            "cltv": [10.0, 20.0],
            "tenure_months": [3, 4],
            "cancel_page_visits_30d": [0, 5],
        }
    )
    numeric, categorical = feature_columns(df, cfg)
    feats = set(numeric) | set(categorical)
    assert "cancel_page_visits_30d" not in feats
    assert "tenure_months" in feats


def test_cli_records_the_drop(tmp_path: Path) -> None:
    p = _template(tmp_path)
    result = CliRunner().invoke(
        app, ["exclude-columns", "--config", str(p), "--add", "cancel_page_visits_30d"]
    )
    assert result.exit_code == 0
    assert load_config(p).columns.exclude_columns == ["cancel_page_visits_30d"]


def test_cli_nothing_to_do_exits_nonzero(tmp_path: Path) -> None:
    p = _template(tmp_path)
    result = CliRunner().invoke(app, ["exclude-columns", "--config", str(p)])
    assert result.exit_code == 1
