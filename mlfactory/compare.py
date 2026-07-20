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

from . import metrics as m
from .config import ChurnConfig
from .model import MODELS, feature_columns, train_model

# Stability gate (the credit-risk convention): small AUC drop + a stable score distribution.
_MAX_AUC_DROP = 0.05
_MAX_SCORE_PSI = 0.2


def compare_models(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    config: ChurnConfig,
    models: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Fit each model on ``train_df``, score ``holdout_df``, and return a ranked comparison.

    Each row: held-out AUC / KS / top-decile lift / PR-AUC, the train→holdout ``auc_drop`` and
    ``ks_drop``, the ``score_psi`` (train→holdout), and a ``stable`` flag. Sorted by held-out AUC.
    """
    models = models or list(MODELS)
    cols = config.columns
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

        rows.append(
            {
                "model": model,
                "holdout_auc": ho_auc,
                "holdout_ks": ho_ks,
                "holdout_lift": m.top_decile_lift(y_ho, p_ho),
                "holdout_pr_auc": m.average_precision(y_ho, p_ho),
                "auc_drop": auc_drop,
                "ks_drop": round(tr_ks - ho_ks, 4),
                "score_psi": score_psi,
                "stable": bool(auc_drop < _MAX_AUC_DROP and score_psi < _MAX_SCORE_PSI),
            }
        )

    rows.sort(key=lambda r: r["holdout_auc"], reverse=True)
    return rows
