"""Tests for train/val/test splitting + the leakage guard + the split-manifest (S6)."""

from __future__ import annotations

import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.split import SplitManifest, SplitError, split_dataset

SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
}


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_accounts=1200, n_months=12, seed=4)


def _cfg(schema=SCHEMA):
    return ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": schema})


def test_time_split_is_time_ordered(panel):
    train, val, test, manifest = split_dataset(panel, _cfg(), strategy="time")
    dc = "observation_month"
    assert train[dc].max() < val[dc].min() < val[dc].max() < test[dc].min()
    assert manifest.leakage.time_ordered is True
    assert manifest.leakage.status == "ok"
    assert manifest.time_windows is not None
    # All three non-empty and covering the panel.
    assert len(train) and len(val) and len(test)
    assert len(train) + len(val) + len(test) == len(panel)


def test_grouped_split_disjoint_accounts(panel):
    train, val, test, manifest = split_dataset(panel, _cfg(), strategy="grouped", seed=1)
    assert manifest.leakage.account_overlap == 0
    assert set(train["account_id"]).isdisjoint(set(test["account_id"]))
    assert manifest.leakage.status == "ok"


def test_random_split_flags_entity_leakage(panel):
    _, _, _, manifest = split_dataset(panel, _cfg(), strategy="random", seed=1)
    # Same account lands in train AND test → the guard warns.
    assert manifest.leakage.account_overlap > 0
    assert manifest.leakage.status == "warn"


def test_row_disjoint_always(panel):
    for strat in ("time", "grouped", "random"):
        _, _, _, manifest = split_dataset(panel, _cfg(), strategy=strat)
        assert manifest.leakage.row_disjoint is True


def test_ratios_approximate(panel):
    train, val, test, _ = split_dataset(panel, _cfg(), strategy="random", seed=2)
    n = len(panel)
    assert abs(len(train) / n - 0.6) < 0.05
    assert abs(len(val) / n - 0.2) < 0.05
    assert abs(len(test) / n - 0.2) < 0.05


def test_time_split_needs_date_col(panel):
    snapshot_cfg = _cfg({"id_col": "account_id", "target_col": "churn_next_30d"})
    with pytest.raises(SplitError, match="date_col"):
        split_dataset(panel, snapshot_cfg, strategy="time")


def test_unknown_strategy_raises(panel):
    with pytest.raises(SplitError, match="unknown strategy"):
        split_dataset(panel, _cfg(), strategy="sideways")


def test_manifest_has_lineage_and_roundtrips(panel, tmp_path):
    _, _, _, manifest = split_dataset(panel, _cfg(), strategy="time")
    assert manifest.artifact == "split-manifest"
    assert manifest.parent_sha256 and len(manifest.parent_sha256) == 64
    p = tmp_path / "split-manifest.json"
    manifest.write_json(p)
    reloaded = SplitManifest.model_validate_json(p.read_text())
    assert reloaded == manifest


def test_grouped_random_deterministic(panel):
    a = split_dataset(panel, _cfg(), strategy="random", seed=9)[3]
    b = split_dataset(panel, _cfg(), strategy="random", seed=9)[3]
    assert a == b
