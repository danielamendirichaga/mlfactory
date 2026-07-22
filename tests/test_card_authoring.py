"""Slice S5 — the model card is *authored*, not just generated.

Accumulated gate/EDA caveats (`config.decisions.caveats`) ride `train` into the card, and DS-authored
sections (`config.decisions.card`: intended use / out of scope / known failure modes / sign-off) render
when present. Defaults add nothing, so a card without authoring looks exactly as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.compute.model import train_model
from mlfactory.config import CardDecisions, ChurnConfig, DecisionRecord
from mlfactory.model_card import gen_model_card


def _mc(**over: object) -> dict:
    base: dict = {
        "model_family": "logistic",
        "n_features": 3,
        "features": ["a", "b", "c"],
        "caveats": [],
    }
    base.update(over)
    return base


def test_authored_sections_render_when_provided() -> None:
    authored = {
        "intended_use": "Rank accounts for the retention team.",
        "out_of_scope": "Not for pricing decisions.",
        "known_failure_modes": ["degrades on brand-new plan tiers"],
        "sign_off": "Reviewed by DS, 2026-07-21.",
    }
    md = gen_model_card(_mc(), authored=authored)
    assert "## Intended Use" in md and "retention team" in md
    assert "## Out of Scope" in md and "pricing" in md
    assert "degrades on brand-new plan tiers" in md  # a failure mode, in Limitations
    assert "## Sign-off" in md and "Reviewed by DS" in md


def test_no_authored_sections_by_default() -> None:
    md = gen_model_card(_mc())
    for heading in ("## Intended Use", "## Out of Scope", "## Sign-off"):
        assert heading not in md


def test_card_caveats_render_in_limitations() -> None:
    assert "engineer or drop signup_month" in gen_model_card(
        _mc(caveats=["engineer or drop signup_month"])
    )


def test_card_decisions_default_empty() -> None:
    c = CardDecisions()
    assert c.intended_use is None and c.out_of_scope is None and c.sign_off is None
    assert c.known_failure_modes == []
    assert DecisionRecord().card == CardDecisions()


def test_train_propagates_caveats_into_the_card() -> None:
    cfg = ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {"id_col": "id", "target_col": "y", "positive_value": 1, "features": "auto"},
            "decisions": {"caveats": ["prefer isotonic over SMOTE at 10% positive"]},
        }
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(200), "f1": rng.normal(size=200), "y": rng.integers(0, 2, 200)})
    _, card = train_model(df, cfg, model="logistic")
    assert card.caveats == ["prefer isotonic over SMOTE at 10% positive"]
    assert "prefer isotonic over SMOTE" in gen_model_card(card.model_dump())


def test_cli_gen_model_card_reads_authored_from_config(tmp_path: Path) -> None:
    card_json = tmp_path / "m.card.json"
    card_json.write_text(json.dumps(_mc()))
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        "source:\n  kind: synthetic\n"
        "schema:\n  id_col: id\n  target_col: y\n  positive_value: 1\n  features: auto\n"
        "decisions:\n  card:\n    intended_use: For the retention team only.\n"
    )
    out = tmp_path / "card.md"
    result = CliRunner().invoke(
        app,
        ["gen-model-card", "--card", str(card_json), "--config", str(cfg), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "## Intended Use" in out.read_text() and "retention team" in out.read_text()
