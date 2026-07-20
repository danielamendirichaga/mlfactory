"""The deterministic feature-transform registry — 8 ``(fit, apply)`` pairs.

**Fit-on-train / apply-outward** is the leakage-safe invariant: stateful transforms
(``standard_scaler``, ``one_hot``, ``impute``, ``target_encoding``) learn their params on the
TRAIN split only, then apply those params outward to val/test. ``target_encoding`` additionally
**CV-folds on train** (a train row never sees its own target) while val/test get the full-train map.

Each transform is a small class exposing ``fit(train, spec) -> params``, ``apply(df, spec, params)``,
``apply_train(train, spec, params)`` (defaults to ``apply``; overridden for CV-folding), and
``produced(spec, params) -> list[str]`` (the columns it adds). numpy/pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfactory.artifacts.schemas import FeatureTransform


class TransformError(ValueError):
    """Raised when a transform cannot be fit or applied (bad arity, missing column, degenerate)."""


def _one(spec: FeatureTransform) -> str:
    if len(spec.inputs) != 1:
        raise TransformError(f"{spec.type} expects exactly 1 input, got {spec.inputs}")
    return spec.inputs[0]


def _require(df: pd.DataFrame, col: str, spec: FeatureTransform) -> None:
    if col not in df.columns:
        raise TransformError(f"{spec.type}: input column {col!r} not in frame")


def _native(v: object) -> object:
    return v.item() if hasattr(v, "item") else v


def _binary_target(s: pd.Series, spec: FeatureTransform) -> np.ndarray:
    """Binarize the target to {0,1} floats via ``params.positive_value`` (default 1)."""
    return (s == spec.params.get("positive_value", 1)).astype(float).to_numpy()


class Transform:
    """Base: fit learns params on train; apply uses them; apply_train defaults to apply."""

    @staticmethod
    def fit(train: pd.DataFrame, spec: FeatureTransform) -> dict:
        return {}

    @staticmethod
    def apply(df: pd.DataFrame, spec: FeatureTransform, params: dict) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def apply_train(cls, train: pd.DataFrame, spec: FeatureTransform, params: dict) -> pd.DataFrame:
        return cls.apply(train, spec, params)

    @staticmethod
    def produced(spec: FeatureTransform, params: dict) -> list[str]:
        return [spec.output_column] if spec.output_column else list(spec.output_columns or [])


class DropColumns(Transform):
    @staticmethod
    def apply(df, spec, params):
        return df.drop(columns=[c for c in spec.inputs if c in df.columns])

    @staticmethod
    def produced(spec, params):
        return []


class LogTransform(Transform):
    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        eps = float(spec.params.get("epsilon", 0.0))
        out = spec.output_column or f"{col}_log"
        df = df.copy()
        df[out] = np.log1p(df[col].astype(float) + eps)
        return df

    @staticmethod
    def produced(spec, params):
        return [spec.output_column or f"{_one(spec)}_log"]


class StandardScaler(Transform):
    @staticmethod
    def fit(train, spec):
        col = _one(spec)
        _require(train, col, spec)
        std = float(train[col].astype(float).std(ddof=0))
        return {"mean": float(train[col].astype(float).mean()), "std": std if std > 0 else 1.0}

    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        out = spec.output_column or f"{col}_scaled"
        df = df.copy()
        df[out] = (df[col].astype(float) - params["mean"]) / params["std"]
        return df

    @staticmethod
    def produced(spec, params):
        return [spec.output_column or f"{_one(spec)}_scaled"]


class OneHot(Transform):
    @staticmethod
    def fit(train, spec):
        col = _one(spec)
        _require(train, col, spec)
        return {"categories": sorted(train[col].dropna().astype(str).unique().tolist())}

    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        df = df.copy()
        s = df[col].astype(str)
        for cat in params["categories"]:
            df[f"{col}_{cat}"] = (s == cat).astype("int64")  # unseen category → all zeros
        if spec.params.get("drop_source", True):
            df = df.drop(columns=[col])
        return df

    @staticmethod
    def produced(spec, params):
        col = _one(spec)
        return [f"{col}_{cat}" for cat in params["categories"]]


class Impute(Transform):
    @staticmethod
    def fit(train, spec):
        col = _one(spec)
        _require(train, col, spec)
        s = train[col]
        default = "median" if pd.api.types.is_numeric_dtype(s) else "mode"
        strategy = spec.params.get("strategy", default)
        if strategy == "median":
            fill = float(s.astype(float).median())
        elif strategy == "mean":
            fill = float(s.astype(float).mean())
        elif strategy == "mode":
            m = s.mode(dropna=True)
            fill = _native(m.iloc[0]) if len(m) else spec.params.get("fill_value", 0)
        elif strategy == "constant":
            fill = spec.params.get("fill_value", 0)
        else:
            raise TransformError(f"impute: unknown strategy {strategy!r}")
        return {"strategy": strategy, "fill_value": fill}

    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        out = spec.output_column or col  # in place by default
        df = df.copy()
        df[out] = df[col].fillna(params["fill_value"])
        return df

    @staticmethod
    def produced(spec, params):
        return [spec.output_column or _one(spec)]


class DateParts(Transform):
    _PARTS = ("year", "month", "day", "dow")
    _EXTRACT = {
        "year": lambda dt: dt.dt.year,
        "month": lambda dt: dt.dt.month,
        "day": lambda dt: dt.dt.day,
        "dow": lambda dt: dt.dt.dayofweek,
    }

    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        df = df.copy()
        dt = pd.to_datetime(df[col])
        for part in spec.params.get("parts", list(DateParts._PARTS)):
            if part not in DateParts._EXTRACT:
                raise TransformError(f"date_parts: unknown part {part!r}")
            df[f"{col}_{part}"] = DateParts._EXTRACT[part](dt).astype("int64")
        if spec.params.get("drop_source", True):
            df = df.drop(columns=[col])
        return df

    @staticmethod
    def produced(spec, params):
        col = _one(spec)
        return [f"{col}_{p}" for p in spec.params.get("parts", list(DateParts._PARTS))]


class TemporalDiff(Transform):
    @staticmethod
    def apply(df, spec, params):
        if len(spec.inputs) != 2:
            raise TransformError(
                f"temporal_diff expects 2 inputs (later, earlier), got {spec.inputs}"
            )
        later, earlier = spec.inputs
        _require(df, later, spec)
        _require(df, earlier, spec)
        out = spec.output_column or f"{later}_minus_{earlier}"
        delta = pd.to_datetime(df[later]) - pd.to_datetime(df[earlier])
        unit = spec.params.get("unit", "days")
        df = df.copy()
        if unit == "days":
            df[out] = delta.dt.total_seconds() / 86400.0
        elif unit == "seconds":
            df[out] = delta.dt.total_seconds()
        else:
            raise TransformError(f"temporal_diff: unknown unit {unit!r}")
        return df

    @staticmethod
    def produced(spec, params):
        later, earlier = spec.inputs[0], spec.inputs[1]
        return [spec.output_column or f"{later}_minus_{earlier}"]


class TargetEncoding(Transform):
    @staticmethod
    def _target(train, spec):
        target = spec.params.get("target")
        if not target:
            raise TransformError("target_encoding requires params.target")
        _require(train, target, spec)
        return target

    @staticmethod
    def _smoothed_map(
        cats: np.ndarray, y: np.ndarray, smoothing: float, global_mean: float
    ) -> dict:
        agg = pd.DataFrame({"c": cats, "y": y}).groupby("c")["y"].agg(["mean", "count"])
        enc = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        return {str(k): float(v) for k, v in enc.to_dict().items()}

    @classmethod
    def fit(cls, train, spec):
        col = _one(spec)
        _require(train, col, spec)
        target = cls._target(train, spec)
        y = _binary_target(train[target], spec)
        smoothing = float(spec.params.get("smoothing", 10.0))
        global_mean = float(y.mean())
        cats = train[col].astype(str).to_numpy()
        return {
            "map": cls._smoothed_map(cats, y, smoothing, global_mean),
            "global_mean": global_mean,
            "smoothing": smoothing,
        }

    @staticmethod
    def apply(df, spec, params):
        col = _one(spec)
        _require(df, col, spec)
        out = spec.output_column or f"{col}_te"
        df = df.copy()
        df[out] = df[col].astype(str).map(params["map"]).fillna(params["global_mean"]).astype(float)
        return df

    @classmethod
    def apply_train(cls, train, spec, params):
        """CV out-of-fold encoding on train so a row never sees its own target (no self-leakage)."""
        col = _one(spec)
        target = cls._target(train, spec)
        out = spec.output_column or f"{col}_te"
        cv = int(spec.params.get("cv_folds", 5))
        smoothing, global_mean = params["smoothing"], params["global_mean"]
        cats = train[col].astype(str).to_numpy()
        y = _binary_target(train[target], spec)
        folds = np.random.default_rng(int(spec.params.get("seed", 42))).integers(0, cv, len(train))
        encoded = np.full(len(train), global_mean, dtype=float)
        for f in range(cv):
            hold = folds == f
            other = ~hold
            if hold.any() and other.any():
                m = cls._smoothed_map(cats[other], y[other], smoothing, global_mean)
                encoded[hold] = pd.Series(cats[hold]).map(m).fillna(global_mean).to_numpy()
        train = train.copy()
        train[out] = encoded
        return train

    @staticmethod
    def produced(spec, params):
        return [spec.output_column or f"{_one(spec)}_te"]


TRANSFORM_REGISTRY: dict[str, type[Transform]] = {
    "drop_columns": DropColumns,
    "log_transform": LogTransform,
    "standard_scaler": StandardScaler,
    "one_hot": OneHot,
    "impute": Impute,
    "date_parts": DateParts,
    "temporal_diff": TemporalDiff,
    "target_encoding": TargetEncoding,
}
