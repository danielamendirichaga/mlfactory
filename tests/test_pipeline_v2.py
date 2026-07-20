"""Capstone v2 end-to-end test (S18) — the uplift/causal arc runs green and pays off.

generate --treatment → risk model + uplift model → Qini eval → risk-vs-uplift contrast →
report with the uplift section, asserting every v2 artifact + that uplift beats risk.
"""

from __future__ import annotations

from mlfactory.config import ChurnConfig
from mlfactory.evaluate import evaluate_model
from mlfactory.generate import make_panel
from mlfactory.model import train_model
from mlfactory.policy import contrast_policies
from mlfactory.qini import evaluate_uplift
from mlfactory.report import build_html
from mlfactory.uplift import train_uplift

SCHEMA = {
    "id_col": "subscriber_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
    "features": [
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
    ],
}


def test_uplift_pipeline_end_to_end():
    cfg = ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SCHEMA})
    df = make_panel(n_subscribers=3000, n_months=12, seed=9, treatment=True)

    # risk + uplift models
    risk, risk_card = train_model(df, cfg, model="logistic", seed=1)
    up, up_card = train_uplift(df, cfg, learner="t", seed=1)
    assert up_card.artifact == "uplift-card"
    assert up_card.tau_recovery_corr is not None and up_card.tau_recovery_corr > 0.25

    # Qini evaluation
    qini = evaluate_uplift(up, df, cfg)
    assert qini.artifact == "qini-report" and qini.qini_coefficient > 0

    # the payoff: targeting by uplift beats targeting by risk (scored on ground truth)
    contrast = contrast_policies(risk, up, df, cfg, offer_cost=3.0, n_offers=1500)
    assert contrast.artifact == "policy-contrast"
    assert contrast.uplift_net_advantage > 0
    assert (
        contrast.strategies["uplift"]["sleeping_dogs_treated"]
        < contrast.strategies["risk"]["sleeping_dogs_treated"]
    )

    # report with the uplift section (v1 charts + qini + uplift-vs-risk)
    ev = evaluate_model(risk, df, cfg)
    html = build_html(
        ev.model_dump(),
        None,
        risk_card.model_dump(),
        qini_report=qini.model_dump(),
        policy_contrast=contrast.model_dump(),
    )
    assert html.startswith("<!doctype html>")
    assert "Uplift — whom does the offer actually change?" in html
    assert html.count("<figure>") == 5

    # every v2 artifact carries lineage
    for art in (up_card, qini, contrast):
        assert art.parent_sha256 and len(art.parent_sha256) == 64
