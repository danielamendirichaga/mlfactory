"""Clean-room churn/risk metric suite — numpy + pandas only (no sklearn).

You cannot unit-test a vibe; you *can* unit-test ``psi(identical) == 0``. This module is the
deterministic, tested compute core the agent delegates every number to.

Conventions
-----------
``y_true`` is binary with 1 = the positive/"churn" class. ``y_score`` is a risk score where
**higher = more likely to churn**. Decile tables are ordered highest-score (riskiest) first.

Contents
--------
* Discrimination: :func:`ks_table` (decile KS), :func:`roc_auc`, :func:`average_precision`.
* Targeting: :func:`gain_table`, :func:`top_decile_lift`.
* Rank stability: :func:`rank_order_breaks`.
* Drift: :func:`psi` (frozen reference edges).
* Threshold metrics: :func:`precision_recall_f1`, :func:`log_loss`.
* Calibration: :func:`calibration_table`, :func:`expected_calibration_error`.

The decile-table KS / ROB are the banking convention (via ``pandas.qcut``), *not*
``scipy.stats.ks_2samp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "KSResult",
    "ks_table",
    "psi",
    "rank_order_breaks",
    "gain_table",
    "top_decile_lift",
    "roc_auc",
    "average_precision",
    "precision_recall_f1",
    "log_loss",
    "calibration_table",
    "expected_calibration_error",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean_pair(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays; drop rows where either value is NaN."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: {y.shape} vs {s.shape}")
    mask = ~(np.isnan(y) | np.isnan(s))
    return y[mask], s[mask]


def _decile_frame(y: np.ndarray, s: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Quantile deciles of the score, aggregated per bin, ordered top-score first."""
    df = pd.DataFrame({"y": y, "s": s})
    df["bin"] = pd.qcut(df["s"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"),
        n_bad=("y", "sum"),
        score_mean=("s", "mean"),
        score_min=("s", "min"),
        score_max=("s", "max"),
    )
    g["n_good"] = g["n"] - g["n_bad"]
    g["bad_rate"] = g["n_bad"] / g["n"]
    return g.sort_values("score_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# KS (decile-table)
# --------------------------------------------------------------------------- #
@dataclass
class KSResult:
    ks: float
    table: list[dict] = field(default_factory=list)
    n_bins: int = 0


def ks_table(y_true, y_score, n_bins: int = 10) -> KSResult:
    """Decile-table KS = max gap between cumulative-bad and cumulative-good curves."""
    y, s = _clean_pair(y_true, y_score)
    if y.size == 0:
        return KSResult(ks=0.0, table=[], n_bins=0)
    g = _decile_frame(y, s, n_bins)
    total_bad, total_good = float(g["n_bad"].sum()), float(g["n_good"].sum())
    if total_bad == 0.0 or total_good == 0.0:
        return KSResult(ks=0.0, table=[], n_bins=int(len(g)))
    g["cum_frac_bad"] = g["n_bad"].cumsum() / total_bad
    g["cum_frac_good"] = g["n_good"].cumsum() / total_good
    g["ks_gap"] = g["cum_frac_bad"] - g["cum_frac_good"]
    table = [
        {
            "decile": i + 1,
            "n": int(r.n),
            "n_bad": int(r.n_bad),
            "bad_rate": round(float(r.bad_rate), 4),
            "cum_frac_bad": round(float(r.cum_frac_bad), 4),
            "cum_frac_good": round(float(r.cum_frac_good), 4),
            "ks_gap": round(float(r.ks_gap), 4),
        }
        for i, r in enumerate(g.itertuples(index=False))
    ]
    return KSResult(ks=round(float(g["ks_gap"].max()), 4), table=table, n_bins=int(len(g)))


# --------------------------------------------------------------------------- #
# PSI (frozen reference edges)
# --------------------------------------------------------------------------- #
def psi(expected, actual, n_bins: int = 10, epsilon: float = 1e-6) -> float:
    """Population Stability Index with edges *frozen* on ``expected``.

    ``PSI = sum((a% - e%) * ln(a% / e%))``. Identical distributions → ~0; a shifted one grows.
    Freezing the reference edges is the key correctness fix over re-binning each period.
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[~np.isnan(exp)]
    act = act[~np.isnan(act)]
    if exp.size == 0 or act.size == 0:
        return 0.0
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp_frac = np.clip(np.histogram(exp, bins=edges)[0] / exp.size, epsilon, None)
    act_frac = np.clip(np.histogram(act, bins=edges)[0] / act.size, epsilon, None)
    return float(np.sum((act_frac - exp_frac) * np.log(act_frac / exp_frac)))


# --------------------------------------------------------------------------- #
# rank-order stability
# --------------------------------------------------------------------------- #
def rank_order_breaks(y_true, y_score, n_bins: int = 10) -> int:
    """Count monotonicity violations in the decile bad-rate (top-score → bottom)."""
    y, s = _clean_pair(y_true, y_score)
    if y.size == 0:
        return 0
    rates = _decile_frame(y, s, n_bins)["bad_rate"].to_numpy()
    if rates.size < 2:
        return 0
    return int(np.sum(rates[:-1] < rates[1:]))


# --------------------------------------------------------------------------- #
# targeting: gain / lift
# --------------------------------------------------------------------------- #
def gain_table(y_true, y_score, n_bins: int = 10) -> list[dict]:
    """Per-decile capture + lift (top-score first). ``lift = bad_rate / base_rate``."""
    y, s = _clean_pair(y_true, y_score)
    if y.size == 0 or y.sum() == 0:
        return []
    g = _decile_frame(y, s, n_bins)
    base = float(y.mean())
    total_bad = float(g["n_bad"].sum())
    g["cum_capture"] = g["n_bad"].cumsum() / total_bad
    g["cum_pop"] = g["n"].cumsum() / float(g["n"].sum())
    return [
        {
            "decile": i + 1,
            "n": int(r.n),
            "n_bad": int(r.n_bad),
            "bad_rate": round(float(r.bad_rate), 4),
            "lift": round(float(r.bad_rate / base), 3),
            "cum_capture": round(float(r.cum_capture), 4),
            "cum_pop": round(float(r.cum_pop), 4),
        }
        for i, r in enumerate(g.itertuples(index=False))
    ]


def top_decile_lift(y_true, y_score) -> float:
    """Positive rate in the top 10% by score, divided by the overall positive rate."""
    y, s = _clean_pair(y_true, y_score)
    base = float(y.mean()) if y.size else 0.0
    if y.size == 0 or base == 0.0:
        return 0.0
    k = max(1, int(round(0.1 * y.size)))
    top = y[np.argsort(-s)[:k]]
    return round(float(top.mean() / base), 3)


# --------------------------------------------------------------------------- #
# AUCs
# --------------------------------------------------------------------------- #
def roc_auc(y_true, y_score) -> float:
    """ROC-AUC via the rank (Mann–Whitney) formula; tie-safe using average ranks."""
    y, s = _clean_pair(y_true, y_score)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return round(float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def average_precision(y_true, y_score) -> float:
    """PR-AUC (average precision): area under the precision-recall curve."""
    y, s = _clean_pair(y_true, y_score)
    total_pos = float(y.sum())
    if y.size == 0 or total_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")  # stable
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return round(float(np.sum((recall - recall_prev) * precision)), 4)


# --------------------------------------------------------------------------- #
# threshold metrics
# --------------------------------------------------------------------------- #
def precision_recall_f1(y_true, y_score, threshold: float = 0.5) -> dict:
    """Precision / recall / F1 for ``y_score >= threshold``."""
    y, s = _clean_pair(y_true, y_score)
    pred = s >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def log_loss(y_true, y_prob, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log loss."""
    y, p = _clean_pair(y_true, y_prob)
    if y.size == 0:
        return 0.0
    p = np.clip(p, eps, 1 - eps)
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def calibration_table(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Reliability table: per equal-width prob bin, mean predicted vs observed rate."""
    y, p = _clean_pair(y_true, y_prob)
    if y.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b + 1,
                "n": int(m.sum()),
                "mean_pred": round(float(p[m].mean()), 4),
                "obs_rate": round(float(y[m].mean()), 4),
            }
        )
    return rows


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: weighted mean |predicted − observed| over the reliability bins."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    total = sum(r["n"] for r in table)
    if not total:
        return 0.0
    return round(float(sum(r["n"] / total * abs(r["mean_pred"] - r["obs_rate"]) for r in table)), 4)
