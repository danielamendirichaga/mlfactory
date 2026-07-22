"""Train/val/test splitting — time-aware (default), grouped, or random — with a leakage guard.

The panel-specific hazard: a **random row-wise** split scatters the same ``account_id`` across
train and test, so the model memorises individuals and scores beautifully — then collapses in
production. So:

* **time** (default) — earliest cohorts → train, latest → test (out-of-time). Accounts
  *legitimately* span the boundary (that mirrors deployment); what must never overlap is a
  account-*month* row.
* **grouped** — every row of a account lands in one split (answers the cold-start question).
* **random** — row-wise; kept *because it's the tempting-wrong one*. The guard detects and
  **reports** the resulting entity leakage.

Emits a :class:`SplitManifest` artifact (with lineage) describing exactly how it split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional

from mlfactory.artifacts import ArtifactBase, content_hash
from mlfactory.config import ChurnConfig


class SplitError(ValueError):
    """Raised when a split cannot be performed (e.g. time split without a date column)."""


class SplitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int
    positive_rate: float


class LeakageCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_disjoint: bool  # no (id, date) row in two splits
    time_ordered: Optional[bool]  # train < val < test in time (None if not a time split)
    account_overlap: int  # accounts appearing in both train and test
    status: Literal["ok", "warn"]


class SplitManifest(ArtifactBase):
    artifact: Literal["split-manifest"] = "split-manifest"
    strategy: Literal["time", "grouped", "random"]
    seed: int
    ratios: dict[str, float]
    train: SplitInfo
    val: SplitInfo
    test: SplitInfo
    time_windows: Optional[dict[str, list[str]]] = None
    leakage: LeakageCheck


def _info(split: pd.DataFrame, target: str, positive) -> SplitInfo:
    rate = float((split[target] == positive).mean()) if len(split) else 0.0
    return SplitInfo(rows=len(split), positive_rate=round(rate, 4))


def _time_split(df: pd.DataFrame, date_col: str, r_train: float, r_val: float):
    dates = sorted(df[date_col].unique())
    n = len(dates)
    if n < 3:
        raise SplitError(f"time split needs ≥3 cohorts, got {n}")
    n_train = max(1, round(n * r_train))
    n_val = max(1, round(n * r_val))
    if n_train + n_val >= n:  # keep at least one cohort for test
        n_val = max(1, n - n_train - 1)
    train_d, val_d, test_d = (
        dates[:n_train],
        dates[n_train : n_train + n_val],
        dates[n_train + n_val :],
    )
    masks = (df[date_col].isin(train_d), df[date_col].isin(val_d), df[date_col].isin(test_d))
    windows = {
        name: [str(pd.Timestamp(min(ds)).date()), str(pd.Timestamp(max(ds)).date())]
        for name, ds in (("train", train_d), ("val", val_d), ("test", test_d))
    }
    return (df[m] for m in masks), windows


def _bucketed(df: pd.DataFrame, key: np.ndarray, r_train: float, r_val: float):
    train = df[key < r_train]
    val = df[(key >= r_train) & (key < r_train + r_val)]
    test = df[key >= r_train + r_val]
    return train, val, test


def _stratified_bucketed(df: pd.DataFrame, y: np.ndarray, r_train: float, r_val: float, seed: int):
    """Random split that preserves the class balance across train/val/test (stratify on the target).

    Each class is ranked on its own random key and spread evenly across [0,1), so ``_bucketed`` hands
    ~``r_train`` of *every* class to train, etc. — deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    key = np.empty(len(df))
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        ranks = rng.random(len(pos)).argsort().argsort()
        key[pos] = (ranks + 0.5) / max(len(pos), 1)
    return _bucketed(df, key, r_train, r_val)


def split_dataset(
    df: pd.DataFrame,
    config: ChurnConfig,
    strategy: str = "time",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Split ``df`` into train/val/test and return them plus a :class:`SplitManifest`."""
    cols = config.columns
    df = df.reset_index(drop=True)
    r_train, r_val, _ = ratios
    windows = None
    time_ordered: Optional[bool] = None

    if strategy == "time":
        if cols.date_col is None or cols.date_col not in df.columns:
            raise SplitError(
                "time split requires a date_col — use --strategy random (or grouped) for snapshot data"
            )
        (train, val, test), windows = _time_split(df, cols.date_col, r_train, r_val)
        time_ordered = bool(
            train[cols.date_col].max() < val[cols.date_col].min()
            and val[cols.date_col].max() < test[cols.date_col].min()
        )
    elif strategy == "grouped":
        ids = df[cols.id_col].unique()
        id_map = dict(zip(ids, np.random.default_rng(seed).random(len(ids))))
        train, val, test = _bucketed(df, df[cols.id_col].map(id_map).to_numpy(), r_train, r_val)
    elif strategy == "random":
        # stratify on the target so the class balance is preserved across train/val/test
        y = (df[cols.target_col] == cols.positive_value).to_numpy()
        train, val, test = _stratified_bucketed(df, y, r_train, r_val, seed)
    else:
        raise SplitError(f"unknown strategy {strategy!r} (use time | grouped | random)")

    # --- leakage guard ---
    idx = [set(train.index), set(val.index), set(test.index)]
    row_disjoint = not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    overlap = len(set(train[cols.id_col]) & set(test[cols.id_col]))
    status: Literal["ok", "warn"] = "warn" if (strategy == "random" and overlap > 0) else "ok"
    leakage = LeakageCheck(
        row_disjoint=row_disjoint,
        time_ordered=time_ordered,
        account_overlap=overlap,
        status=status,
    )

    manifest = SplitManifest(
        strategy=strategy,  # type: ignore[arg-type]
        seed=seed,
        ratios={"train": r_train, "val": r_val, "test": round(1 - r_train - r_val, 4)},
        train=_info(train, cols.target_col, cols.positive_value),
        val=_info(val, cols.target_col, cols.positive_value),
        test=_info(test, cols.target_col, cols.positive_value),
        time_windows=windows,
        leakage=leakage,
        parent_sha256=content_hash(df),
    )
    return train, val, test, manifest
