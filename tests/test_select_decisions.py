"""Slice S3 — train & select read the decision record.

`compare` ranks (and `recommend_model` selects) on `config.decisions.modeling.primary_metric` with the
record's stability bars, and `train` honors the recorded imbalance / calibration / tune regime — instead
of hardcoded AUC, hardcoded `0.05/0.2`, and hidden `--smote`/`--calibrate` flags. Defaults reproduce
today's behavior (primary_metric=auc, bars 0.05/0.2, regime off).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.compute.compare import compare_models
from mlfactory.config import ChurnConfig
from mlfactory.recommend import recommend_model


def _cfg(**modeling: object) -> ChurnConfig:
    base: dict = {
        "source": {"kind": "synthetic"},
        "schema": {"id_col": "id", "target_col": "y", "positive_value": 1, "features": "auto"},
    }
    if modeling:
        base["decisions"] = {"modeling": modeling}
    return ChurnConfig.model_validate(base)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * f1 - 0.6 * f2)))).astype(int)
    return pd.DataFrame({"id": range(n), "f1": f1, "f2": f2, "y": y})


def test_compare_rows_carry_primary_and_default_to_auc() -> None:
    rows = compare_models(_frame(seed=1), _frame(seed=2), _cfg(), models=["logistic", "rf"])
    assert all(r["primary_metric"] == "auc" for r in rows)
    assert all(r["primary"] == r["holdout_auc"] for r in rows)  # default metric
    assert rows == sorted(rows, key=lambda r: r["primary"], reverse=True)


def test_primary_metric_changes_the_ranking_key() -> None:
    rows = compare_models(
        _frame(seed=1), _frame(seed=2), _cfg(primary_metric="pr_auc"), models=["logistic", "rf"]
    )
    assert all(r["primary"] == r["holdout_pr_auc"] for r in rows)
    assert rows == sorted(rows, key=lambda r: r["holdout_pr_auc"], reverse=True)


def test_stability_bars_come_from_the_record() -> None:
    tr, ho = _frame(seed=1), _frame(seed=2)
    loose = compare_models(
        tr, ho, _cfg(max_auc_drop=1.0, max_score_psi=10.0), models=["logistic", "rf"]
    )
    tight = compare_models(
        tr, ho, _cfg(max_auc_drop=0.0, max_score_psi=0.0), models=["logistic", "rf"]
    )
    assert all(r["stable"] for r in loose)  # everything stable under loose bars
    assert not any(r["stable"] for r in tight)  # nothing stable under impossible bars


def test_recommend_model_selects_on_primary_not_auc() -> None:
    # A wins on pr_auc, B wins on auc; with primary=pr_auc the pick is A.
    rows = [
        {
            "model": "A",
            "holdout_auc": 0.70,
            "auc_drop": 0.01,
            "stable": True,
            "primary_metric": "pr_auc",
            "primary": 0.40,
        },
        {
            "model": "B",
            "holdout_auc": 0.80,
            "auc_drop": 0.01,
            "stable": True,
            "primary_metric": "pr_auc",
            "primary": 0.30,
        },
    ]
    rec = recommend_model(rows)
    assert rec.action["model"] == "A"
    assert "pr_auc" in rec.rationale


def test_recommend_model_is_backward_compatible_without_primary() -> None:
    rows = [{"model": "X", "holdout_auc": 0.7, "auc_drop": 0.01, "stable": True}]
    assert recommend_model(rows).action["model"] == "X"  # falls back to holdout_auc


def test_train_cli_honors_the_recorded_regime(tmp_path: Path) -> None:
    frame = _frame(seed=3)
    train_pq = tmp_path / "train.parquet"
    frame.to_parquet(train_pq, index=False)
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        "source:\n  kind: synthetic\n"
        "schema:\n  id_col: id\n  target_col: y\n  positive_value: 1\n  features: auto\n"
        "decisions:\n  modeling:\n    calibrate: true\n"
    )
    out = tmp_path / "m.pkl"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--train",
            str(train_pq),
            "--config",
            str(cfg),
            "--model",
            "logistic",
            "--model-out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    card = json.loads((tmp_path / "m.card.json").read_text())
    assert card["calibrated"] is True  # read from config.decisions.modeling, no CLI flag passed
