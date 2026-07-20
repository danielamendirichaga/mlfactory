"""Tests for uplift evaluation (S16) — Qini coefficient ordering, curve, deciles, recovery."""

from __future__ import annotations

import numpy as np
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.generate import make_panel
from mlfactory.qini import (
    QiniError,
    QiniReport,
    evaluate_uplift,
    qini_coefficient,
    uplift_by_decile,
)
from mlfactory.uplift import train_uplift

FEATURES = [
    "tenure_months",
    "monthly_price",
    "watch_hours_30d",
    "days_since_last_watch",
    "watch_hours_trend",
    "support_tickets_30d",
    "payment_failures_30d",
    "promo_months_left",
    "on_promo",
    "plan_tier",
    "region",
]


def _cfg():
    return ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {
                "id_col": "subscriber_id",
                "target_col": "churn_next_30d",
                "date_col": "observation_month",
                "value_col": "cltv",
                "features": FEATURES,
            },
        }
    )


@pytest.fixture(scope="module")
def scored():
    df = make_panel(n_subscribers=4000, n_months=12, seed=9, treatment=True)
    cfg = _cfg()
    model, _ = train_uplift(df, cfg, learner="t", seed=1)
    retained = 1 - (df["churn_next_30d"] == 1).to_numpy().astype(int)
    treated = df["treated"].to_numpy().astype(int)
    pred = model.predict_uplift(df)
    return df, cfg, model, retained, treated, pred


def test_qini_ordering_oracle_beats_model_beats_random(scored):
    df, _, _, retained, treated, pred = scored
    true = df["true_uplift"].to_numpy()
    q_true = qini_coefficient(retained, treated, true)
    q_pred = qini_coefficient(retained, treated, pred)
    q_none = qini_coefficient(retained, treated, np.zeros(len(df)))  # no targeting
    assert q_true > q_pred > q_none  # the oracle targets best; the model beats no-targeting
    assert q_pred > 0  # and beats random in absolute terms


def test_uplift_deciles_are_monotone_and_top_beats_bottom(scored):
    _, _, _, retained, treated, pred = scored
    rows = uplift_by_decile(retained, treated, pred)
    assert len(rows) == 10 and rows[0]["decile"] == 1
    # predicted uplift decreases by decile by construction (1 = best-targeted)
    preds = [r["mean_pred"] for r in rows]
    assert preds == sorted(preds, reverse=True)
    # and the *observed* uplift is higher at the top than the bottom
    top3 = np.mean([rows[i]["obs_uplift"] for i in range(3)])
    bot3 = np.mean([rows[i]["obs_uplift"] for i in range(7, 10)])
    assert top3 > bot3


def test_evaluate_uplift_report(scored):
    df, cfg, model, *_ = scored
    report = evaluate_uplift(model, df, cfg)
    assert report.artifact == "qini-report"
    assert report.n_treated + report.n_control == report.n_rows
    assert report.qini_coefficient > 0
    assert report.tau_recovery_corr is not None and report.tau_recovery_corr > 0.25
    assert len(report.qini_curve) > 5
    assert report.qini_curve[0]["frac"] == 0.0 and report.qini_curve[-1]["frac"] == pytest.approx(
        1.0
    )


def test_requires_treatment_column(scored):
    _, cfg, model, *_ = scored
    plain = make_panel(n_subscribers=300, n_months=6, seed=2)  # no treated column
    with pytest.raises(QiniError, match="treated"):
        evaluate_uplift(model, plain, cfg)


def test_report_lineage_and_roundtrip(scored, tmp_path):
    df, cfg, model, *_ = scored
    report = evaluate_uplift(model, df, cfg)
    assert report.parent_sha256 and len(report.parent_sha256) == 64
    p = tmp_path / "qini-report.json"
    report.write_json(p)
    assert QiniReport.model_validate_json(p.read_text()) == report
