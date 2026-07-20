"""Stage-4 feature engineering — run the transform recipe, fit-on-train / apply-outward.

Given a list of :class:`FeatureTransform` specs and the split frames, :func:`engineer_features`
fits each stateful transform on TRAIN, applies it outward to val/test, threads the frames through
the sequence, records the learned params (for audit + replay), and enforces the **model-ready
postcondition** (every produced column numeric/boolean, no nulls/NaN/inf). :func:`build_feature_spec`
assembles the ``feature-spec`` artifact.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from mlfactory.artifacts.base import Verification, schema_hash
from mlfactory.artifacts.schemas import DatasetOutput, FeatureSpecArtifact, FeatureTransform
from mlfactory.compute.engineer_transforms import TRANSFORM_REGISTRY, TransformError


class FeatureEngineeringError(ValueError):
    """Raised when a transform fails or the engineered output is not model-ready."""


def _validate_model_ready(df: pd.DataFrame, columns: list[str]) -> None:
    """Every produced feature column must be numeric/boolean with no nulls/NaN/inf."""
    for c in columns:
        if c not in df.columns:
            continue  # legitimately removed by a later drop_columns
        s = df[c]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            raise FeatureEngineeringError(f"model-ready violation: {c!r} is not numeric/boolean")
        if bool(pd.isna(s).any()):
            raise FeatureEngineeringError(f"model-ready violation: {c!r} has nulls")
        arr = s.to_numpy()
        if np.issubdtype(arr.dtype, np.floating) and not bool(np.isfinite(arr).all()):
            raise FeatureEngineeringError(f"model-ready violation: {c!r} has inf/NaN")


def engineer_features(
    transforms: list[FeatureTransform],
    train: pd.DataFrame,
    *,
    val: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, Optional[pd.DataFrame]], dict, list[str]]:
    """Apply the transform sequence fit-on-train / apply-outward.

    Returns ``(frames, fit_params, produced)`` where ``frames`` has keys ``train``/``val``/``test``
    (``None`` when not supplied), ``fit_params`` maps each transform id → its learned params, and
    ``produced`` lists the feature columns the transforms added.
    """
    train_df = train.reset_index(drop=True).copy()
    others: dict[str, Optional[pd.DataFrame]] = {
        "val": None if val is None else val.reset_index(drop=True).copy(),
        "test": None if test is None else test.reset_index(drop=True).copy(),
    }
    fit_params: dict = {}
    produced: list[str] = []
    for spec in transforms:
        transform = TRANSFORM_REGISTRY.get(spec.type)
        if transform is None:
            raise FeatureEngineeringError(f"unknown transform type {spec.type!r}")
        try:
            params = transform.fit(train_df, spec)
            train_df = transform.apply_train(train_df, spec, params)
            for split in ("val", "test"):
                frame = others[split]
                if frame is not None:
                    others[split] = transform.apply(frame, spec, params)
        except TransformError as exc:
            raise FeatureEngineeringError(f"transform {spec.id} ({spec.type}): {exc}") from exc
        fit_params[str(spec.id)] = params
        for col in transform.produced(spec, params):
            if col not in produced:
                produced.append(col)

    _validate_model_ready(train_df, produced)
    for split in ("val", "test"):
        frame = others[split]
        if frame is not None:
            _validate_model_ready(frame, produced)
    return {"train": train_df, "val": others["val"], "test": others["test"]}, fit_params, produced


def build_feature_spec(
    transforms: list[FeatureTransform],
    engineered_train: pd.DataFrame,
    fit_params: dict,
    *,
    output_path: str = "",
    parent_sha256: Optional[str] = None,
) -> FeatureSpecArtifact:
    """Assemble the ``feature-spec`` artifact from the engineered train frame + learned params."""
    return FeatureSpecArtifact(
        verification=Verification(status="passed", method="deterministic_script"),
        transforms=transforms,
        fit_params=fit_params,
        output=DatasetOutput(
            path=output_path,
            row_count=len(engineered_train),
            schema_hash=schema_hash(engineered_train),
        ),
        parent_sha256=parent_sha256,
    )
