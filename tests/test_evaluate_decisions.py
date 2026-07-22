"""Slice S4 — evaluate + ship read the decision record.

The operating threshold, the segments to slice, and the ship acceptance criteria come from
`config.decisions.evaluation` instead of a silent `0.5`, hardcoded `plan_tier/region`, and a baked-in
`0.65/0.10` bar. Defaults reproduce today's behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfactory.compute.evaluate import evaluate_model
from mlfactory.compute.model import train_model
from mlfactory.config import ChurnConfig
from mlfactory.recommend import recommend_ship


def _cfg(evaluation: dict | None = None) -> ChurnConfig:
    base: dict = {
        "source": {"kind": "synthetic"},
        "schema": {"id_col": "id", "target_col": "y", "positive_value": 1, "features": "auto"},
    }
    if evaluation:
        base["decisions"] = {"evaluation": evaluation}
    return ChurnConfig.model_validate(base)


def _frame(n: int = 300, seed: int = 0, seg: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * f1 - 0.6 * f2)))).astype(int)
    df = pd.DataFrame({"id": range(n), "f1": f1, "f2": f2, "y": y})
    if seg:
        df["seg"] = ["a", "b"] * (n // 2)
    return df


def _fit(cfg: ChurnConfig, seg: bool = False) -> object:
    est, _ = train_model(_frame(seed=1, seg=seg), cfg, model="logistic")
    return est


def test_threshold_defaults_from_the_record() -> None:
    cfg = _cfg({"threshold": 0.2})
    assert evaluate_model(_fit(cfg), _frame(seed=2), cfg).threshold == 0.2


def test_threshold_default_is_half_without_a_record() -> None:
    cfg = _cfg()
    assert evaluate_model(_fit(cfg), _frame(seed=2), cfg).threshold == 0.5


def test_explicit_threshold_overrides_the_record() -> None:
    cfg = _cfg({"threshold": 0.2})
    assert evaluate_model(_fit(cfg), _frame(seed=2), cfg, threshold=0.8).threshold == 0.8


def test_segment_cols_come_from_the_record() -> None:
    cfg = _cfg({"segment_cols": ["seg"]})
    rep = evaluate_model(_fit(cfg, seg=True), _frame(seed=2, seg=True), cfg)
    assert set(rep.segments.keys()) == {"seg"}


def test_recommend_ship_honors_the_recorded_criteria() -> None:
    report = {"metrics": {"auc": 0.70, "ece": 0.05}}
    assert recommend_ship(report, min_auc=0.65, max_ece=0.10).action["ship"] is True
    assert (
        recommend_ship(report, min_auc=0.75, max_ece=0.10).action["ship"] is False
    )  # stricter floor
    assert (
        recommend_ship(report, min_auc=0.65, max_ece=0.01).action["ship"] is False
    )  # stricter ECE bar


def test_record_decision_sets_a_list_valued_segment(tmp_path) -> None:
    from mlfactory.config import CONFIG_TEMPLATE, load_config, set_decision

    p = tmp_path / "churn.yaml"
    p.write_text(CONFIG_TEMPLATE)
    set_decision(p, "evaluation.segment_cols", '["plan_tier"]')  # JSON list, not a bare string
    assert load_config(p).decisions.evaluation.segment_cols == ["plan_tier"]
