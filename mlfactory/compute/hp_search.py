"""Optuna TPE hyperparameter search — seeded, deterministic, over the model menu.

A more efficient search than a grid: Optuna's **TPE sampler (seeded)** proposes trials over a
per-family space, each scored by cross-validated ROC-AUC in the leakage-safe pipeline, and the winner
is refit on the full train split. Same ``seed`` → same trials → same winner (reproducible + auditable).

Supported families: ``logistic``, ``rf``, ``xgboost``, ``hist_gbm`` (the pruned ``tree`` is tuned by
cost-complexity pruning, not Optuna).
"""

from __future__ import annotations

from typing import Any

from mlfactory.compute.model import ModelError, feature_columns
from mlfactory.config import ChurnConfig

OPTUNA_FAMILIES = ("logistic", "rf", "xgboost", "hist_gbm")


def _space(trial: Any, model: str) -> dict:
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
        }
    if model == "hist_gbm":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }
    if model == "logistic":
        return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}
    raise ModelError(f"no Optuna search space for model {model!r} (use {OPTUNA_FAMILIES})")


def _make(model: str, params: dict, seed: int):
    if model == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            **params,
        )
    if model == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, **params)
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            solver="saga", l1_ratio=1.0, max_iter=2000, random_state=seed, **params
        )
    raise ModelError(f"no Optuna estimator for model {model!r}")


def optuna_search(
    train_df,
    config: ChurnConfig,
    model: str,
    *,
    n_trials: int = 30,
    seed: int = 42,
) -> tuple[Any, dict, float]:
    """Seeded Optuna TPE search for ``model`` on ``train_df``. Returns (fitted pipeline, best params,
    best CV-AUC). Deterministic: same ``seed`` reproduces the same winner."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline as SkPipeline

    from mlfactory.compute.model import _preprocessor

    if model not in OPTUNA_FAMILIES:
        raise ModelError(f"Optuna search supports {OPTUNA_FAMILIES}, not {model!r}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    numeric, categorical = feature_columns(train_df, config)
    cols = config.columns
    x = train_df[numeric + categorical]
    y = (train_df[cols.target_col] == cols.positive_value).astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        pipe = SkPipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, model)),
                ("model", _make(model, _space(trial, model), seed)),
            ]
        )
        return float(cross_val_score(pipe, x, y, scoring="roc_auc", cv=cv, n_jobs=1).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = SkPipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, model)),
            ("model", _make(model, study.best_params, seed)),
        ]
    )
    best.fit(x, y)
    return best, dict(study.best_params), float(study.best_value)
