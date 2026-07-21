"""Slice S2a — an engineered feature-spec actually reaches the model.

`train_model(..., engineered=True)` trains on the model-ready output of `engineer-features`, passing
its features through (impute leftover nulls + one-hot any surviving categoricals, but NO re-scaling —
the recipe owns that). Before this, `train` only ever read the raw split, so a recipe changed nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfactory.compute.model import ModelError, train_model
from mlfactory.config import ChurnConfig


def _cfg() -> ChurnConfig:
    return ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {"id_col": "id", "target_col": "y", "positive_value": 1, "features": "auto"},
        }
    )


def _frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * f1 - 0.6 * f2)))).astype(int)
    return pd.DataFrame({"id": range(n), "f1": f1, "f2": f2, "y": y})


def test_engineered_fits_and_records_the_flag() -> None:
    est, card = train_model(_frame(), _cfg(), model="logistic", engineered=True)
    assert card.engineered is True
    assert card.n_features == 2  # f1, f2 (id/y are reserved)
    assert est.predict_proba(_frame()[["f1", "f2"]]).shape == (240, 2)


def test_engineered_mode_does_not_rescale_numerics() -> None:
    est, _ = train_model(_frame(), _cfg(), model="logistic", engineered=True)
    num_tr = dict(est.named_steps["preprocess"].named_transformers_)["num"]
    assert "scale" not in [name for name, _ in num_tr.steps]  # recipe owns scaling


def test_engineered_handles_surviving_categorical_and_nulls() -> None:
    df = _frame()
    df["region"] = ["a", "b"] * (len(df) // 2)  # a raw categorical the recipe left behind
    df.loc[0, "f1"] = np.nan  # a leftover null in a numeric
    est, card = train_model(df, _cfg(), model="logistic", engineered=True)
    assert card.engineered is True
    est.predict_proba(df[["f1", "f2", "region"]])  # one-hot + impute handled them, no error


@pytest.mark.parametrize("kw", [{"tune": True}, {"optuna": True}, {"early_stopping": True}])
def test_engineered_rejects_search_combos(kw: dict) -> None:
    model = "xgboost" if "early_stopping" in kw else "logistic"
    with pytest.raises(ModelError):
        train_model(_frame(), _cfg(), model=model, engineered=True, **kw)


def test_engineered_evaluate_roundtrip() -> None:
    from mlfactory.compute.evaluate import evaluate_model

    est, _ = train_model(_frame(seed=1), _cfg(), model="logistic", engineered=True)
    report = evaluate_model(est, _frame(seed=2), _cfg())
    assert 0.0 <= report.metrics["auc"] <= 1.0
    assert report.n_rows == 240


def test_recipe_output_reaches_the_model() -> None:
    """The thesis of the slice: the columns engineer-features produced end up in the model."""
    from mlfactory.artifacts.schemas import FeatureTransform
    from mlfactory.compute.engineer import engineer_features

    tr = _frame(seed=1)
    tr["f1"] = np.abs(tr["f1"]) + 0.1  # positive → log-transformable
    spec = [FeatureTransform(id=1, name="log f1", type="log_transform", inputs=["f1"], params={})]
    frames, _, produced = engineer_features(spec, tr)
    _, card = train_model(frames["train"], _cfg(), model="logistic", engineered=True)
    assert card.engineered
    assert produced and set(produced).issubset(set(card.features))
