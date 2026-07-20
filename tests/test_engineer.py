"""Tests for Stage-4 feature engineering — the transform registry + fit-on-train/apply-outward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfactory.artifacts.schemas import FeatureTransform
from mlfactory.compute.engineer import (
    FeatureEngineeringError,
    build_feature_spec,
    engineer_features,
)


def _t(**kw) -> FeatureTransform:
    kw.setdefault("id", 1)
    kw.setdefault("name", kw["type"])
    return FeatureTransform.model_validate(kw)


# --- fit-on-train / apply-outward (the leakage-safe invariant) ------------ #
def test_standard_scaler_fits_on_train_only():
    train = pd.DataFrame({"x": [0.0, 10.0]})  # mean 5, std 5
    test = pd.DataFrame({"x": [5.0]})
    spec = _t(type="standard_scaler", inputs=["x"], output_column="xs")
    frames, params, produced = engineer_features([spec], train, test=test)
    assert params["1"]["mean"] == 5.0
    # the test point 5 is scaled with the TRAIN mean/std → 0.0 (not with its own stats)
    assert frames["test"]["xs"].iloc[0] == pytest.approx(0.0)
    assert produced == ["xs"]


def test_one_hot_unseen_category_is_all_zero():
    train = pd.DataFrame({"g": ["a", "b"]})
    test = pd.DataFrame({"g": ["c"]})  # unseen category
    frames, _, produced = engineer_features([_t(type="one_hot", inputs=["g"])], train, test=test)
    assert set(produced) == {"g_a", "g_b"}
    assert frames["test"][["g_a", "g_b"]].iloc[0].tolist() == [0, 0]  # unseen → all zeros
    assert "g" not in frames["train"].columns  # drop_source default


def test_impute_uses_train_fill():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan]})  # train median = 2.0
    test = pd.DataFrame({"x": [np.nan]})
    frames, params, _ = engineer_features([_t(type="impute", inputs=["x"])], train, test=test)
    assert params["1"]["fill_value"] == 2.0
    assert frames["test"]["x"].iloc[0] == 2.0


# --- target_encoding: CV-folded on train, full-train map on test ---------- #
def test_target_encoding_no_self_leakage_on_train():
    # A category appearing once in train: its CV out-of-fold encoding cannot come from its own row,
    # so it equals the global mean (with smoothing 0) — proving no self-leakage.
    train = pd.DataFrame({"g": ["solo", "x", "x", "x", "x"], "y": [1, 0, 0, 0, 0]})
    spec = _t(
        type="target_encoding",
        inputs=["g"],
        output_column="g_te",
        params={"target": "y", "smoothing": 0.0, "cv_folds": 5, "seed": 1},
    )
    frames, params, _ = engineer_features([spec], train)
    solo_enc = frames["train"].loc[train["g"] == "solo", "g_te"].iloc[0]
    assert solo_enc == pytest.approx(params["1"]["global_mean"])  # NOT 1.0 (its own target)


def test_target_encoding_test_uses_full_train_map():
    train = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1, 1, 1, 1, 0, 0, 0, 0]})
    test = pd.DataFrame({"g": ["a", "b"], "y": [0, 0]})
    spec = _t(
        type="target_encoding",
        inputs=["g"],
        output_column="g_te",
        params={"target": "y", "smoothing": 0.0},
    )
    frames, _, _ = engineer_features([spec], train, test=test)
    assert frames["test"]["g_te"].tolist() == pytest.approx([1.0, 0.0])  # full-train map


# --- stateless transforms ------------------------------------------------- #
def test_log_date_temporal_drop():
    df = pd.DataFrame(
        {
            "v": [0.0, np.e - 1],
            "d": pd.to_datetime(["2023-01-15", "2023-06-20"]),
            "d0": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "junk": [1, 2],
        }
    )
    specs = [
        _t(id=1, type="log_transform", inputs=["v"], output_column="v_log"),
        _t(
            id=2,
            type="temporal_diff",
            inputs=["d", "d0"],
            output_column="gap",
            params={"unit": "days"},
        ),
        _t(id=3, type="date_parts", inputs=["d"], params={"parts": ["year", "month"]}),
        _t(id=4, type="drop_columns", inputs=["junk"]),
    ]
    out = engineer_features(specs, df)[0]["train"]
    assert out["v_log"].tolist() == pytest.approx([0.0, 1.0])
    assert out["d_year"].tolist() == [2023, 2023] and out["d_month"].tolist() == [1, 6]
    assert out["gap"].tolist() == pytest.approx([14.0, 19.0])
    assert "junk" not in out.columns


# --- model-ready postcondition ------------------------------------------- #
def test_model_ready_rejects_nan_output():
    train = pd.DataFrame({"x": [-2.0, 3.0]})  # log1p(-2) is NaN
    spec = _t(type="log_transform", inputs=["x"], output_column="x_log")
    with pytest.raises(FeatureEngineeringError, match="model-ready"):
        engineer_features([spec], train)


def test_feature_transform_rejects_unknown_type():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        FeatureTransform.model_validate({"id": 1, "name": "x", "type": "nope", "inputs": ["a"]})


# --- the feature-spec artifact validates end-to-end ----------------------- #
def test_feature_spec_artifact_validates(tmp_path):
    from mlfactory.artifacts.validate import validate_artifact

    train = pd.DataFrame({"x": [0.0, 10.0], "y": [1, 0]})
    spec = _t(type="standard_scaler", inputs=["x"], output_column="xs")
    frames, fit_params, _ = engineer_features([spec], train)
    frames["train"].to_parquet(tmp_path / "train.parquet", index=False)
    written = pd.read_parquet(tmp_path / "train.parquet")
    art = build_feature_spec([spec], written, fit_params, output_path="train.parquet")
    art.write_markdown(tmp_path / "feature-spec.md")
    result = validate_artifact(tmp_path / "feature-spec.md", probe_output=True)
    assert result["artifact"] == "feature-spec"


# --- CLI end-to-end ------------------------------------------------------- #
def test_cli_engineer_features(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from mlfactory.cli import app
    from mlfactory.domains.saas.generate import make_panel

    df = make_panel(n_accounts=200, n_months=6, seed=3)
    cut = df["observation_month"].quantile(0.7)
    df[df["observation_month"] < cut].to_parquet(tmp_path / "train.parquet", index=False)
    df[df["observation_month"] >= cut].to_parquet(tmp_path / "test.parquet", index=False)
    (tmp_path / "spec.yaml").write_text(
        yaml.safe_dump(
            {
                "transforms": [
                    {
                        "id": 1,
                        "name": "scale tenure",
                        "type": "standard_scaler",
                        "inputs": ["tenure_months"],
                        "output_column": "tenure_scaled",
                    },
                    {"id": 2, "name": "onehot plan", "type": "one_hot", "inputs": ["plan_tier"]},
                ]
            }
        )
    )
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "engineer-features",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "out" / "train.parquet").exists()
    assert (tmp_path / "out" / "feature-spec.md").exists()
    # the emitted feature-spec artifact validates (schema + on-disk probe)
    v = runner.invoke(
        app, ["validate-artifact", str(tmp_path / "out" / "feature-spec.md"), "--probe-output"]
    )
    assert v.exit_code == 0, v.output
