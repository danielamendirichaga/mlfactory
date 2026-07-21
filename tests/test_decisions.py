"""Slice S0 — the decision record: gates WRITE it, stages READ it.

The load-bearing test is `test_defaults_match_current_hardcoded_behavior`: every default on the
`DecisionRecord` is asserted equal to the value the pipeline hardcodes today, so wiring a later stage
to read `config.decisions.*` (S3/S4/S6) is behavior-preserving until a gate overrides it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.config import CONFIG_TEMPLATE, ConfigError, DecisionRecord, load_config, set_decision


def _template(tmp_path: Path) -> Path:
    p = tmp_path / "churn.yaml"
    p.write_text(CONFIG_TEMPLATE)
    return p


def test_defaults_match_current_hardcoded_behavior() -> None:
    from mlfactory import recommend
    from mlfactory.compute import evaluate
    from mlfactory.compute.compare import _MAX_AUC_DROP, _MAX_SCORE_PSI
    from mlfactory.domains.saas import monitor

    d = DecisionRecord()
    # stability gate (compare.py module constants)
    assert d.modeling.max_auc_drop == _MAX_AUC_DROP
    assert d.modeling.max_score_psi == _MAX_SCORE_PSI
    # today's selection metric is held-out AUC
    assert d.modeling.primary_metric == "auc"
    # evaluate operating threshold + ship bar + drift bar (their function defaults)
    assert (
        d.evaluation.threshold
        == inspect.signature(evaluate.evaluate_model).parameters["threshold"].default
    )
    ship = inspect.signature(recommend.recommend_ship).parameters
    assert d.evaluation.min_auc == ship["min_auc"].default
    assert d.evaluation.max_ece == ship["max_ece"].default
    assert (
        d.monitoring.drift_threshold
        == inspect.signature(monitor.monitor_drift).parameters["threshold"].default
    )


def test_config_without_decisions_block_uses_defaults(tmp_path: Path) -> None:
    assert load_config(_template(tmp_path)).decisions == DecisionRecord()


def test_config_with_partial_decisions_block(tmp_path: Path) -> None:
    p = _template(tmp_path)
    p.write_text(p.read_text() + "\ndecisions:\n  evaluation:\n    threshold: 0.2\n")
    cfg = load_config(p)
    assert cfg.decisions.evaluation.threshold == 0.2
    assert cfg.decisions.modeling.primary_metric == "auc"  # untouched → default


def test_set_decision_persists_coerces_and_keeps_comments(tmp_path: Path) -> None:
    p = _template(tmp_path)
    rec = set_decision(p, "evaluation.threshold", "0.3")  # string → float
    assert rec.evaluation.threshold == 0.3
    assert load_config(p).decisions.evaluation.threshold == 0.3
    assert "# mlfactory config" in p.read_text()  # source/schema comments survive


def test_set_decision_validates_a_literal(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_decision(p, "modeling.primary_metric", "pr_auc")
    assert load_config(p).decisions.modeling.primary_metric == "pr_auc"


def test_set_decision_unknown_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        set_decision(_template(tmp_path), "modeling.nope", "1")


def test_set_decision_invalid_value_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        set_decision(_template(tmp_path), "modeling.primary_metric", "not_a_metric")


def test_set_decision_updates_in_place_single_block(tmp_path: Path) -> None:
    p = _template(tmp_path)
    set_decision(p, "evaluation.threshold", "0.3")
    set_decision(p, "evaluation.threshold", "0.4")
    assert load_config(p).decisions.evaluation.threshold == 0.4
    headers = [ln for ln in p.read_text().splitlines() if ln.startswith("decisions:")]
    assert len(headers) == 1  # block replaced in place, not duplicated


def test_cli_record_and_show(tmp_path: Path) -> None:
    p = _template(tmp_path)
    runner = CliRunner()
    r = runner.invoke(
        app, ["record-decision", "--config", str(p), "--key", "policy.offer_cost", "--value", "8"]
    )
    assert r.exit_code == 0
    assert load_config(p).decisions.policy.offer_cost == 8.0
    shown = runner.invoke(app, ["decisions", "--config", str(p), "--json"])
    assert shown.exit_code == 0
    assert '"offer_cost": 8.0' in shown.stdout
