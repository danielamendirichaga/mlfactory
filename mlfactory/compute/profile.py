"""Per-column data profiling — the EDA numbers the agent reasons over.

:func:`profile_frame` returns one record per column: its role, null rate, cardinality,
numeric summary stats, and — for numeric features — a correlation to the **binarized** target
(``target == positive_value``, the same rule ``train``/``evaluate`` use, so it works for string
labels like ``"Yes"/"No"`` too). That target correlation is what surfaces the planted leakage
trap (``cancel_page_visits_30d`` shows an extreme correlation); :func:`high_corr_features`
turns that into a soft "possible leakage" hint the agent can act on.

Roles for the id / date / target columns come from the config (not guessed), which matters
in panel data where ``account_id`` is *not* mostly-unique. numpy/pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfactory.config import ChurnConfig

__all__ = ["profile_frame", "infer_role", "high_corr_features"]


def infer_role(series: pd.Series) -> str:
    """Coarse role from dtype: datetime / categorical / numeric."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _role_for(col: str, config: ChurnConfig, series: pd.Series) -> str:
    cols = config.columns
    if col == cols.id_col:
        return "id"
    if col == cols.date_col:
        return "datetime"
    if col == cols.target_col:
        return "target"
    return infer_role(series)


def _safe_corr(feature: pd.Series, target: pd.Series) -> float | None:
    """Pearson correlation on rows where both are present; None if undefined."""
    mask = feature.notna() & target.notna()
    if int(mask.sum()) < 2:
        return None
    f, t = feature[mask].to_numpy(), target[mask].to_numpy()
    if f.std() == 0 or t.std() == 0:
        return None
    return round(float(np.corrcoef(f, t)[0, 1]), 4)


def profile_frame(df: pd.DataFrame, config: ChurnConfig) -> list[dict]:
    """Profile every column of ``df``. One dict per column.

    Always includes ``column``, ``role``, ``null_rate``, ``n_unique``. Numeric columns add
    ``min/max/mean/std/q25/q50/q75``; numeric *features* also add ``target_corr``.
    """
    target = config.columns.target_col
    # Binarize via positive_value (as train/evaluate do) so the leakage correlation works for
    # string labels ("Yes"/"No"), not just 0/1 targets.
    target_binary = (
        (df[target] == config.columns.positive_value).astype(float)
        if target in df.columns
        else None
    )
    n_rows = len(df)
    records: list[dict] = []

    for col in df.columns:
        s = df[col]
        role = _role_for(col, config, s)
        rec: dict = {
            "column": col,
            "role": role,
            "null_rate": round(float(s.isna().mean()), 4) if n_rows else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if role == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            valid = num.dropna()
            if len(valid):
                rec.update(
                    min=round(float(valid.min()), 2),
                    max=round(float(valid.max()), 2),
                    mean=round(float(valid.mean()), 2),
                    std=round(float(valid.std(ddof=1)), 2) if len(valid) > 1 else 0.0,
                    q25=round(float(valid.quantile(0.25)), 2),
                    q50=round(float(valid.quantile(0.50)), 2),
                    q75=round(float(valid.quantile(0.75)), 2),
                )
            if target_binary is not None and col != target:
                rec["target_corr"] = _safe_corr(num, target_binary)
        records.append(rec)

    return records


def high_corr_features(records: list[dict], threshold: float = 0.5) -> list[tuple[str, float]]:
    """Numeric features whose |target_corr| ≥ threshold — a soft leakage hint."""
    out = [
        (r["column"], r["target_corr"])
        for r in records
        if r.get("target_corr") is not None and abs(r["target_corr"]) >= threshold
    ]
    return sorted(out, key=lambda x: -abs(x[1]))


def scan_leakage(records: list[dict], config: ChurnConfig) -> list[dict]:
    """Tier the target correlations into structured leakage risks — the deterministic substrate the
    EDA leakage-scanner subagent wraps, then reasons over for the posterior/derived cases it can't see.

    Tiers (blueprint §8.3): ``|corr| > 0.99`` → perfect_predictor (drop); ``0.9 ≤ |corr| < 0.99`` →
    near_perfect (inspect — verify it is observable at prediction time). Genuine drivers stay below.
    """
    target = config.columns.target_col
    risks: list[dict] = []
    for r in records:
        corr = r.get("target_corr")
        if corr is None or r["column"] == target:
            continue
        a = abs(corr)
        if a > 0.99:
            kind, rec = "perfect_predictor", "drop"
            reason = f"|corr|={a:.3f} > 0.99 — computed from the target, contains it, or encodes posterior state"
        elif a >= 0.9:
            kind, rec = "near_perfect", "inspect"
            reason = f"|corr|={a:.3f} in [0.9, 0.99) — verify it is observable at prediction time"
        else:
            continue
        risks.append(
            {
                "column": r["column"],
                "target": target,
                "strength": round(float(corr), 4),
                "kind": kind,
                "recommendation": rec,
                "reason": reason,
            }
        )
    return sorted(risks, key=lambda x: -abs(x["strength"]))
