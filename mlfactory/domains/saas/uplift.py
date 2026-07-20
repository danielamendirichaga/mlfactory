"""Uplift modeling — target *persuadables*, not just high risk.

Two meta-learners over the v1 model stack (`model.py`), so uplift reuses the same leakage-safe
pipeline and estimators — no causal library:

* **S-learner** — one churn model with ``treated`` as a feature;
  ``τ̂(x) = P(churn | x, control) − P(churn | x, treat)``.
* **T-learner** — two churn models (control / treated); ``τ̂(x) = f₀(x) − f₁(x)``.

Uplift is the **reduction in churn probability** the offer causes (positive = the offer helps;
negative = a sleeping dog). Requires the A/B columns from ``make_panel(treatment=True)``. When the
synthetic ``true_uplift`` is present, the card also reports how well the learner *recovers* it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline

from mlfactory.artifacts import ArtifactBase, content_hash
from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import ORACLE_COLS, TREATMENT_COL
from mlfactory.compute.model import MODELS, _estimator, _preprocessor, feature_columns

UPLIFT_LEARNERS = ("s", "t")


class UpliftError(ValueError):
    """Raised when an uplift model cannot be fit (no treatment column, bad learner, …)."""


class UpliftCard(ArtifactBase):
    artifact: Literal["uplift-card"] = "uplift-card"
    learner: str
    base_model: str
    n_train: int
    n_treated: int
    n_control: int
    ate_hat: float  # mean predicted uplift on train
    tau_recovery_corr: Optional[float] = None  # corr(τ̂, true τ) when the oracle is present
    features: list[str]


def _target(df: pd.DataFrame, config: ChurnConfig) -> np.ndarray:
    cols = config.columns
    return (df[cols.target_col] == cols.positive_value).astype(int).to_numpy()


def _churn_pipe(
    numeric: list[str], categorical: list[str], base_model: str, seed: int
) -> SkPipeline:
    return SkPipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, base_model)),
            ("model", _estimator(base_model, seed, tune=False)),
        ]
    )


class UpliftModel:
    """A fitted S- or T-learner exposing ``predict_uplift``."""

    def __init__(self, learner: str, base_model: str, seed: int) -> None:
        self.learner = learner
        self.base_model = base_model
        self.seed = seed
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self._s: Any = None
        self._t0: Any = None
        self._t1: Any = None

    def fit(self, df: pd.DataFrame, config: ChurnConfig) -> UpliftModel:
        self.numeric, self.categorical = feature_columns(df, config)  # excludes treated + oracle
        y = _target(df, config)
        treated = df[TREATMENT_COL].to_numpy()
        if self.learner == "s":
            num_s = [*self.numeric, TREATMENT_COL]
            self._s = _churn_pipe(num_s, self.categorical, self.base_model, self.seed)
            self._s.fit(df[num_s + self.categorical], y)
        else:  # t-learner
            cols = self.numeric + self.categorical
            self._t0 = _churn_pipe(self.numeric, self.categorical, self.base_model, self.seed)
            self._t1 = _churn_pipe(self.numeric, self.categorical, self.base_model, self.seed)
            self._t0.fit(df[cols][treated == 0], y[treated == 0])
            self._t1.fit(df[cols][treated == 1], y[treated == 1])
        return self

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        """Estimated churn-probability reduction from treating each row (higher = better target)."""
        if self.learner == "s":
            cols = [*self.numeric, TREATMENT_COL, *self.categorical]
            x0, x1 = df.copy(), df.copy()
            x0[TREATMENT_COL] = 0
            x1[TREATMENT_COL] = 1
            p_control = self._s.predict_proba(x0[cols])[:, 1]
            p_treat = self._s.predict_proba(x1[cols])[:, 1]
        else:
            cols = self.numeric + self.categorical
            p_control = self._t0.predict_proba(df[cols])[:, 1]
            p_treat = self._t1.predict_proba(df[cols])[:, 1]
        return p_control - p_treat


def train_uplift(
    df: pd.DataFrame,
    config: ChurnConfig,
    learner: str = "t",
    base_model: str = "logistic",
    seed: int = 42,
) -> tuple[UpliftModel, UpliftCard]:
    """Fit an uplift meta-learner and return it with a lineage-stamped `UpliftCard`."""
    if learner not in UPLIFT_LEARNERS:
        raise UpliftError(f"unknown learner {learner!r} (use {' | '.join(UPLIFT_LEARNERS)})")
    if base_model not in MODELS:
        raise UpliftError(f"unknown base model {base_model!r} (use {' | '.join(MODELS)})")
    if TREATMENT_COL not in df.columns:
        raise UpliftError(
            f"uplift needs a {TREATMENT_COL!r} column — generate with `--treatment` (a randomized A/B test)"
        )

    # A/B ground-truth columns must never be features (defensive for `features: auto` panels);
    # the uplift model stays self-consistent because it stores its fitted feature set.
    config = config.model_copy(deep=True)
    config.columns.exclude_columns = sorted(
        set(config.columns.exclude_columns) | {TREATMENT_COL, *ORACLE_COLS}
    )
    model = UpliftModel(learner, base_model, seed).fit(df, config)
    tau_hat = model.predict_uplift(df)
    treated = df[TREATMENT_COL].to_numpy()

    recovery = None
    if "true_uplift" in df.columns and np.std(tau_hat) > 0:
        recovery = round(float(np.corrcoef(tau_hat, df["true_uplift"].to_numpy())[0, 1]), 4)

    card = UpliftCard(
        learner=learner,
        base_model=base_model,
        n_train=len(df),
        n_treated=int((treated == 1).sum()),
        n_control=int((treated == 0).sum()),
        ate_hat=round(float(tau_hat.mean()), 4),
        tau_recovery_corr=recovery,
        features=model.numeric + model.categorical,
        parent_sha256=content_hash(df),
    )
    return model, card


def save_uplift(model: UpliftModel, path) -> None:
    joblib.dump(model, path)


def load_uplift(path) -> UpliftModel:
    return joblib.load(path)
