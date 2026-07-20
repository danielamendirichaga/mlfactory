"""Drift monitoring — has the input distribution moved enough to warrant a retrain?

:func:`monitor_drift` compares each numeric feature's distribution in the **earliest** cohort
(the reference) against the **latest** cohort, using PSI (frozen on the reference). Features
past the threshold are flagged and, if any drift, a retrain is *recommended* — the DS decides;
mlfactory proposes, never auto-retrains. In snapshot mode (no ``date_col``) it **skips
gracefully**. Emits a ``drift-report`` artifact. Reuses `metrics.psi`.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from mlfactory.compute import metrics as m
from mlfactory.artifacts import ArtifactBase, content_hash
from mlfactory.config import ChurnConfig
from mlfactory.compute.model import feature_columns

_MODERATE, _MAJOR = 0.1, 0.25


class DriftReport(ArtifactBase):
    artifact: Literal["drift-report"] = "drift-report"
    mode: str  # "panel" | "snapshot"
    skipped: bool
    reference: Optional[str] = None
    latest: Optional[str] = None
    threshold: float
    features: list[dict]  # {feature, psi, status}
    drifted: list[str]
    retrain_recommended: bool


def _status(psi: float) -> str:
    return "major" if psi > _MAJOR else "moderate" if psi > _MODERATE else "stable"


def monitor_drift(data: pd.DataFrame, config: ChurnConfig, threshold: float = 0.25) -> DriftReport:
    """Per-feature PSI (earliest → latest cohort) + a retrain recommendation."""
    cols = config.columns
    base = dict(
        threshold=threshold,
        features=[],
        drifted=[],
        retrain_recommended=False,
        parent_sha256=content_hash(data),
    )

    if cols.date_col is None or cols.date_col not in data.columns:
        return DriftReport(mode="snapshot", skipped=True, **base)

    dates = sorted(data[cols.date_col].unique())
    if len(dates) < 2:
        return DriftReport(
            mode="panel", skipped=True, reference=str(pd.Timestamp(dates[0]).date()), **base
        )

    reference, latest = dates[0], dates[-1]
    ref_df = data[data[cols.date_col] == reference]
    cur_df = data[data[cols.date_col] == latest]
    numeric, _ = feature_columns(data, config)

    features: list[dict] = []
    for f in numeric:
        val = round(m.psi(ref_df[f], cur_df[f]), 4)
        features.append({"feature": f, "psi": val, "status": _status(val)})
    features.sort(key=lambda r: -float(r["psi"]))
    drifted = [str(r["feature"]) for r in features if float(r["psi"]) > threshold]

    return DriftReport(
        mode="panel",
        skipped=False,
        reference=str(pd.Timestamp(reference).date()),
        latest=str(pd.Timestamp(latest).date()),
        threshold=threshold,
        features=features,
        drifted=drifted,
        retrain_recommended=len(drifted) > 0,
        parent_sha256=content_hash(data),
    )
