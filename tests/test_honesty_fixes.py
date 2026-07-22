"""#36 — three 'claims-X-does-Y' fixes found by running on real data, each replacing a hardcode
with context-awareness:

1. logistic regularization is a decision (default L1, via sklearn's l1_ratio) — NB: the reported
   "L2 not L1" defect was a false alarm; the original l1_ratio=1.0 was already L1 in sklearn 1.9;
2. the `random` split stratifies on the target (was plain random — a real fix);
3. the model card's provenance caveat is conditional on `source.kind` (was hardcoded 'synthetic' — a real fix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfactory.compute.model import train_model
from mlfactory.compute.split import split_dataset
from mlfactory.config import ChurnConfig
from mlfactory.model_card import gen_model_card


def _cfg(source: dict | None = None, **modeling: object) -> ChurnConfig:
    base: dict = {
        "source": source or {"kind": "synthetic"},
        "schema": {"id_col": "id", "target_col": "y", "positive_value": 1, "features": "auto"},
    }
    if modeling:
        base["decisions"] = {"modeling": modeling}
    return ChurnConfig.model_validate(base)


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * f1 - 0.6 * f2)))).astype(int)
    return pd.DataFrame({"id": range(n), "f1": f1, "f2": f2, "y": y})


# --- Fix 1: logistic regularization is a decision (sklearn's l1_ratio: 1.0 = L1, 0.0 = L2) ---


def test_logistic_default_is_l1() -> None:
    est, card = train_model(_frame(), _cfg(), model="logistic")
    assert est.named_steps["model"].l1_ratio == 1.0  # 1.0 = pure L1 (= the prior behavior)
    assert card.hyperparams.get("l1_ratio") == 1.0  # the card records it accurately


def test_penalty_decision_switches_the_model() -> None:
    est, _ = train_model(_frame(), _cfg(penalty="l2"), model="logistic")
    assert est.named_steps["model"].l1_ratio == 0.0  # l2 → l1_ratio 0.0


# --- Fix 2: the random split stratifies on the target ---


def test_random_split_preserves_class_balance() -> None:
    df = _frame(n=1000, seed=1)
    train, val, test, _ = split_dataset(df, _cfg(), strategy="random", seed=42)
    overall = float((df["y"] == 1).mean())
    for part in (train, val, test):
        assert (
            abs(float((part["y"] == 1).mean()) - overall) < 0.02
        )  # stratified → matched prevalence


# --- Fix 3: card provenance conditional on source.kind ---


def test_card_says_synthetic_only_for_synthetic_source() -> None:
    _, card = train_model(_frame(), _cfg(source={"kind": "synthetic"}), model="logistic")
    assert card.source_kind == "synthetic"
    assert "synthetic B2B SaaS reference domain" in gen_model_card(card.model_dump())


def test_card_does_not_claim_synthetic_for_a_real_source() -> None:
    cfg = _cfg(
        source={"kind": "file", "path": "d.parquet"}
    )  # train reads the frame, not the source
    _, card = train_model(_frame(), cfg, model="logistic")
    assert card.source_kind == "file"
    md = gen_model_card(card.model_dump())
    assert "synthetic B2B SaaS reference domain" not in md
    assert "provided data" in md  # the honest real-data caveat instead
