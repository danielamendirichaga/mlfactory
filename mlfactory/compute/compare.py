"""Model comparison — fit the shortlist, rank on held-out performance *and* stability.

The point isn't "highest AUC wins." A model that peaks on train but drops hard on held-out
data (or whose score distribution shifts) is not the one you deploy. So :func:`compare_models`
reports, per model, held-out performance **and** stability — the train→holdout metric drops
and the score-PSI — and flags which candidates are actually stable. The agent narrates the
trade-off; the DS picks.

Reuses `model.train_model` (fit) and the tested `metrics` core (score) — no new math.
"""

from __future__ import annotations

import pandas as pd

from mlfactory.compute import metrics as m
from mlfactory.config import ChurnConfig
from mlfactory.compute.model import MODELS, feature_columns, train_model

# Stability gate (the credit-risk convention): small AUC drop + a stable score distribution.
# These are the DEFAULTS; the live values come from config.decisions.modeling (epic #17 / S3).
_MAX_AUC_DROP = 0.05
_MAX_SCORE_PSI = 0.2

# primary_metric name (config.decisions.modeling) → the compare-row key that holds it
_METRIC_KEY = {
    "auc": "holdout_auc",
    "pr_auc": "holdout_pr_auc",
    "ks": "holdout_ks",
    "top_decile_lift": "holdout_lift",
}


def compare_models(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    config: ChurnConfig,
    models: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Fit each model on ``train_df``, score ``holdout_df``, and return a ranked comparison.

    Each row: held-out AUC / KS / top-decile lift / PR-AUC, the train→holdout ``auc_drop`` and
    ``ks_drop``, the ``score_psi`` (train→holdout), a ``stable`` flag, and ``primary`` (the value of
    the DS's chosen ``primary_metric``). Rows are sorted by ``primary`` (default: held-out AUC), and
    the stability bars come from ``config.decisions.modeling`` (default: the module constants).
    """
    models = models or list(MODELS)
    cols = config.columns
    dec = config.decisions.modeling
    metric_key = _METRIC_KEY[dec.primary_metric]
    numeric, categorical = feature_columns(train_df, config)
    feats = numeric + categorical
    y_tr = (train_df[cols.target_col] == cols.positive_value).astype(int).to_numpy()
    y_ho = (holdout_df[cols.target_col] == cols.positive_value).astype(int).to_numpy()

    rows: list[dict] = []
    for model in models:
        est, _ = train_model(train_df, config, model=model, seed=seed)
        p_tr = est.predict_proba(train_df[feats])[:, 1]
        p_ho = est.predict_proba(holdout_df[feats])[:, 1]

        tr_auc, ho_auc = m.roc_auc(y_tr, p_tr), m.roc_auc(y_ho, p_ho)
        tr_ks, ho_ks = m.ks_table(y_tr, p_tr).ks, m.ks_table(y_ho, p_ho).ks
        auc_drop = round(tr_auc - ho_auc, 4)
        score_psi = round(m.psi(p_tr, p_ho), 4)

        row = {
            "model": model,
            "holdout_auc": ho_auc,
            "holdout_ks": ho_ks,
            "holdout_lift": m.top_decile_lift(y_ho, p_ho),
            "holdout_pr_auc": m.average_precision(y_ho, p_ho),
            "auc_drop": auc_drop,
            "ks_drop": round(tr_ks - ho_ks, 4),
            "score_psi": score_psi,
            # stability bars from the decision record (defaults = the module constants)
            "stable": bool(auc_drop < dec.max_auc_drop and score_psi < dec.max_score_psi),
        }
        # the DS's chosen primary metric drives ranking + selection (default: auc)
        row["primary_metric"] = dec.primary_metric
        row["primary"] = row[metric_key]
        rows.append(row)

    rows.sort(key=lambda r: r["primary"], reverse=True)
    return rows
