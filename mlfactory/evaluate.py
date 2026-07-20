"""Held-out evaluation — the honest scorecard (train metrics were optimistic).

:func:`evaluate_model` loads no model itself; it takes a fitted estimator, scores a held-out
frame, and reports the full **union metric pack** (discrimination + targeting + threshold
metrics + calibration), **per-segment slices** (where is the model weak?), and an optional
**score-PSI vs. a reference** (has the score distribution drifted?). Emits an ``eval-report``
artifact with lineage. Reuses the tested `metrics` core — no new math.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from . import metrics as m
from .artifacts import ArtifactBase, content_hash
from .config import ChurnConfig
from .model import feature_columns


class EvalReport(ArtifactBase):
    artifact: Literal["eval-report"] = "eval-report"
    n_rows: int
    threshold: float
    metrics: dict
    calibration: list[dict]
    gain: list[dict] = []
    segments: dict
    score_psi: Optional[float] = None


def _auc_or_none(y, p) -> Optional[float]:
    """ROC-AUC, or None when a slice has a single class (keeps the JSON NaN-free)."""
    return m.roc_auc(y, p) if len(set(y.tolist())) == 2 else None


def evaluate_model(
    estimator,
    test_df: pd.DataFrame,
    config: ChurnConfig,
    reference_df: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
    segment_cols: Optional[list[str]] = None,
) -> EvalReport:
    """Score ``test_df`` with ``estimator`` and return a full :class:`EvalReport`."""
    cols = config.columns
    test_df = test_df.reset_index(drop=True)
    if cols.features != "auto":
        missing = [c for c in cols.features if c not in test_df.columns]
        if missing:
            raise ValueError(f"test data is missing feature columns: {missing}")
    numeric, categorical = feature_columns(test_df, config)
    feats = numeric + categorical

    y = (test_df[cols.target_col] == cols.positive_value).astype(int).to_numpy()
    proba = estimator.predict_proba(test_df[feats])[:, 1]

    prf = m.precision_recall_f1(y, proba, threshold)
    metrics = {
        "auc": m.roc_auc(y, proba),
        "pr_auc": m.average_precision(y, proba),
        "ks": m.ks_table(y, proba).ks,
        "rank_order_breaks": m.rank_order_breaks(y, proba),
        "top_decile_lift": m.top_decile_lift(y, proba),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "log_loss": m.log_loss(y, proba),
        "ece": m.expected_calibration_error(y, proba),
    }

    # Per-segment slices — where is the model weak?
    if segment_cols is None:
        segment_cols = [c for c in ("plan_tier", "region") if c in test_df.columns]
    segments: dict = {}
    for col in segment_cols:
        seg: dict = {}
        for level in test_df[col].dropna().unique():
            mask = (test_df[col] == level).to_numpy()
            yy, pp = y[mask], proba[mask]
            seg[str(level)] = {
                "n": int(mask.sum()),
                "churn_rate": round(float(yy.mean()), 4),
                "auc": _auc_or_none(yy, pp),
                "lift": m.top_decile_lift(yy, pp),
            }
        segments[col] = seg

    score_psi: Optional[float] = None
    if reference_df is not None:
        ref = reference_df.reset_index(drop=True)
        ref_proba = estimator.predict_proba(ref[feats])[:, 1]
        score_psi = round(m.psi(ref_proba, proba), 4)

    return EvalReport(
        n_rows=len(test_df),
        threshold=threshold,
        metrics=metrics,
        calibration=m.calibration_table(y, proba),
        gain=m.gain_table(y, proba),
        segments=segments,
        score_psi=score_psi,
        parent_sha256=content_hash(test_df),
    )
