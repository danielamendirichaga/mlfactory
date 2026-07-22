"""Slice S2b — construction transforms (ratio / interaction) + the FE-approach decision.

`ratio` and `interaction` let an EDA-informed recipe *build* signal (usage-per-seat, engagement×recency)
— the highest-value move on a low-|corr| problem — and the FE gate's choice lives in
`config.decisions.features.approach` (default `skip` = train on the raw split, today's behavior).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlfactory.artifacts.schemas import FeatureTransform
from mlfactory.compute.engineer import FeatureEngineeringError, engineer_features
from mlfactory.compute.engineer_transforms import (
    TRANSFORM_REGISTRY,
    Interaction,
    Ratio,
    TransformError,
)
from mlfactory.config import CONFIG_TEMPLATE, ConfigError, DecisionRecord, load_config, set_decision


def test_registry_has_construction_transforms() -> None:
    assert "ratio" in TRANSFORM_REGISTRY and "interaction" in TRANSFORM_REGISTRY


def test_ratio_divides_and_names() -> None:
    df = pd.DataFrame({"usage": [10.0, 20.0], "seats": [2.0, 5.0]})
    spec = FeatureTransform(id=1, name="upS", type="ratio", inputs=["usage", "seats"])
    out = Ratio.apply(df, spec, {})
    assert out["usage_per_seats"].tolist() == [5.0, 4.0]
    assert Ratio.produced(spec, {}) == ["usage_per_seats"]


def test_ratio_zero_denominator_is_filled_and_finite() -> None:
    df = pd.DataFrame({"a": [3.0, 7.0], "b": [0.0, 2.0]})
    spec = FeatureTransform(
        id=1, name="r", type="ratio", inputs=["a", "b"], params={"on_zero": -1.0}
    )
    out = Ratio.apply(df, spec, {})
    assert out["a_per_b"].tolist() == [-1.0, 3.5]  # 3/0 → on_zero, 7/2 → 3.5
    assert bool(np.isfinite(out["a_per_b"]).all())


def test_ratio_requires_two_inputs() -> None:
    spec = FeatureTransform(id=1, name="r", type="ratio", inputs=["a"])
    with pytest.raises(TransformError):
        Ratio.apply(pd.DataFrame({"a": [1.0]}), spec, {})


def test_interaction_multiplies_two_and_three() -> None:
    df = pd.DataFrame({"x": [2.0, 3.0], "y": [4.0, 5.0], "z": [10.0, 1.0]})
    s2 = FeatureTransform(id=1, name="xy", type="interaction", inputs=["x", "y"])
    assert Interaction.apply(df, s2, {})["x_x_y"].tolist() == [8.0, 15.0]
    s3 = FeatureTransform(
        id=2, name="xyz", type="interaction", inputs=["x", "y", "z"], output_column="prod"
    )
    assert Interaction.apply(df, s3, {})["prod"].tolist() == [80.0, 15.0]


def test_interaction_requires_two_inputs() -> None:
    spec = FeatureTransform(id=1, name="i", type="interaction", inputs=["x"])
    with pytest.raises(TransformError):
        Interaction.apply(pd.DataFrame({"x": [1.0]}), spec, {})


def test_construction_through_engineer_features_is_model_ready() -> None:
    df = pd.DataFrame(
        {"usage": [10.0, 20.0, 30.0], "seats": [2.0, 4.0, 5.0], "logins": [1.0, 2.0, 3.0]}
    )
    specs = [
        FeatureTransform(id=1, name="upS", type="ratio", inputs=["usage", "seats"]),
        FeatureTransform(id=2, name="uxl", type="interaction", inputs=["usage", "logins"]),
    ]
    frames, _, produced = engineer_features(specs, df)  # runs the model-ready postcondition
    assert set(produced) == {"usage_per_seats", "usage_x_logins"}
    assert frames["train"]["usage_per_seats"].tolist() == [5.0, 5.0, 6.0]
    assert frames["train"]["usage_x_logins"].tolist() == [10.0, 40.0, 90.0]


def test_ratio_null_input_is_caught_by_model_ready() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 2.0]})
    spec = FeatureTransform(id=1, name="r", type="ratio", inputs=["a", "b"])
    with pytest.raises(FeatureEngineeringError):  # NaN propagates into a produced column
        engineer_features([spec], df)


# --- FeatureDecisions: the FE gate's home in the decision record ---


def _template(tmp_path: Path) -> Path:
    p = tmp_path / "churn.yaml"
    p.write_text(CONFIG_TEMPLATE)
    return p


def test_feature_decision_default_is_skip() -> None:
    d = DecisionRecord()
    assert d.features.approach == "skip"  # = train on the raw split (today's behavior)
    assert d.features.recipe_path is None


def test_record_feature_approach(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_decision(p, "features.approach", "recipe")
    assert load_config(p).decisions.features.approach == "recipe"


def test_bad_feature_approach_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        set_decision(_template(tmp_path), "features.approach", "nonsense")
