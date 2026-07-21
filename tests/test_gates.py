"""Tests for the human-in-the-loop gates substrate — advise --json (structured recommendations)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.domains.saas.generate import make_panel

runner = CliRunner()


def test_cli_advise_json(tmp_path):
    panel = tmp_path / "p.parquet"
    make_panel(n_accounts=400, n_months=8, seed=6).to_parquet(panel, index=False)
    cfg = tmp_path / "churn.yaml"
    cfg.write_text(
        f"source:\n  kind: file\n  path: {panel}\n"
        "schema:\n  id_col: account_id\n  target_col: churn_next_30d\n"
        "  date_col: observation_month\n  value_col: cltv\n"
    )
    r = runner.invoke(app, ["advise", "--config", str(cfg), "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip())
    assert payload["command"] == "advise"
    gates = {rec["gate"] for rec in payload["recommendations"]}
    assert {"features", "split"} <= gates
    # each recommendation is the what / why / action a gate surfaces
    for rec in payload["recommendations"]:
        assert {"gate", "recommendation", "rationale", "action"} <= set(rec)
