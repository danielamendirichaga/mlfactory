"""Uplift evaluation — Qini curve, Qini coefficient, and uplift-by-decile.

How good is the *targeting*? Rank customers by predicted uplift τ̂ and ask: as we treat more
of the top of that list, how many **extra retentions** do we buy versus not targeting? That is
the Qini curve; the area between it and the random diagonal is the Qini coefficient. Everything
is framed on **retention** (``retained = 1 − churn``) so higher = better.

Pure numpy/pandas (the tested-core spirit — no sklearn here). Because the synthetic generator
knows the true τ, the report also states how well the model *recovers* it.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from mlfactory.artifacts import ArtifactBase, content_hash
from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import TREATMENT_COL
from mlfactory.domains.saas.uplift import UpliftModel


class QiniError(ValueError):
    """Raised when uplift cannot be evaluated (e.g. no treatment column)."""


class QiniReport(ArtifactBase):
    artifact: Literal["qini-report"] = "qini-report"
    n_rows: int
    n_treated: int
    n_control: int
    qini_coefficient: float
    qini_curve: list[dict]  # [{frac, qini, random}]
    uplift_deciles: list[dict]  # [{decile, n, obs_uplift, mean_pred}]
    tau_recovery_corr: Optional[float] = None


def _qini_points(
    retained: np.ndarray, treated: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental retentions as we descend the score-ranked list."""
    order = np.argsort(-score, kind="mergesort")  # best-targeted first, stable
    r = retained[order].astype(float)
    t = treated[order].astype(bool)
    n_t = np.cumsum(t)
    n_c = np.cumsum(~t)
    r_t = np.cumsum(r * t)
    r_c = np.cumsum(r * ~t)
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = r_t - r_c * np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0.0)
    frac = np.arange(1, len(retained) + 1) / len(retained)
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], qini])


def qini_coefficient(retained: np.ndarray, treated: np.ndarray, score: np.ndarray) -> float:
    """Area between the model's Qini curve and the random diagonal (>0 beats random)."""
    x, q = _qini_points(retained, treated, score)
    area_model = float(np.sum((x[1:] - x[:-1]) * (q[1:] + q[:-1]) / 2.0))
    area_random = 0.5 * float(q[-1])  # triangle under the (0,0)→(1, q_end) diagonal
    return round(area_model - area_random, 4)


def qini_curve(
    retained: np.ndarray, treated: np.ndarray, score: np.ndarray, n_points: int = 40
) -> list[dict]:
    x, q = _qini_points(retained, treated, score)
    q_end = float(q[-1])
    idx = np.unique(np.linspace(0, len(x) - 1, min(n_points, len(x))).astype(int))
    return [
        {
            "frac": round(float(x[i]), 4),
            "qini": round(float(q[i]), 2),
            "random": round(x[i] * q_end, 2),
        }
        for i in idx
    ]


def uplift_by_decile(
    retained: np.ndarray, treated: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Observed uplift (treated − control retention) per predicted-uplift decile (1 = top)."""
    d = pd.DataFrame({"r": retained, "t": treated, "s": score})
    ranks = d["s"].rank(method="first", ascending=False)
    d["decile"] = (pd.qcut(ranks, n_bins, labels=False) + 1).astype(int)
    rows = []
    for dec, g in d.groupby("decile"):
        gt, gc = g[g["t"] == 1]["r"], g[g["t"] == 0]["r"]
        obs = float(gt.mean() - gc.mean()) if len(gt) and len(gc) else float("nan")
        rows.append(
            {
                "decile": int(dec),
                "n": int(len(g)),
                "obs_uplift": None if np.isnan(obs) else round(obs, 4),
                "mean_pred": round(float(g["s"].mean()), 4),
            }
        )
    return rows


def evaluate_uplift(
    model: UpliftModel, df: pd.DataFrame, config: ChurnConfig, n_bins: int = 10
) -> QiniReport:
    """Score ``df`` with the uplift model and package a `QiniReport`."""
    if TREATMENT_COL not in df.columns:
        raise QiniError(
            f"uplift evaluation needs a {TREATMENT_COL!r} column (a randomized A/B panel)"
        )
    cols = config.columns
    churn = (df[cols.target_col] == cols.positive_value).to_numpy().astype(int)
    retained = 1 - churn
    treated = df[TREATMENT_COL].to_numpy().astype(int)
    score = model.predict_uplift(df)

    recovery = None
    if "true_uplift" in df.columns and float(np.std(score)) > 0:
        recovery = round(float(np.corrcoef(score, df["true_uplift"].to_numpy())[0, 1]), 4)

    return QiniReport(
        n_rows=len(df),
        n_treated=int((treated == 1).sum()),
        n_control=int((treated == 0).sum()),
        qini_coefficient=qini_coefficient(retained, treated, score),
        qini_curve=qini_curve(retained, treated, score),
        uplift_deciles=uplift_by_decile(retained, treated, score, n_bins=n_bins),
        tau_recovery_corr=recovery,
        parent_sha256=content_hash(df),
    )
