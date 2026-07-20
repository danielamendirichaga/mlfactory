"""Capstone end-to-end test (S13) — the whole v1 pipeline runs green and emits every artifact.

generate → split → train → compare → evaluate → simulate-policy → report → monitor,
exercised through the module functions on tiny synthetic data.
"""

from __future__ import annotations

from mlfactory.compute.compare import compare_models
from mlfactory.config import ChurnConfig
from mlfactory.compute.evaluate import evaluate_model
from mlfactory.domains.saas.generate import make_panel
from mlfactory.compute.model import load_model, save_model, train_model
from mlfactory.domains.saas.monitor import monitor_drift
from mlfactory.domains.saas.policy import simulate_policy
from mlfactory.report import build_html
from mlfactory.compute.split import split_dataset

SCHEMA = {
    "id_col": "account_id",
    "target_col": "churn_next_30d",
    "date_col": "observation_month",
    "value_col": "cltv",
    "features": [
        "tenure_months",
        "mrr",
        "product_usage_hours_30d",
        "days_since_last_login",
        "support_tickets_30d",
        "plan_tier",
        "region",
    ],
}


def test_full_pipeline_end_to_end(tmp_path):
    cfg = ChurnConfig.model_validate({"source": {"kind": "synthetic"}, "schema": SCHEMA})
    df = make_panel(n_accounts=1000, n_months=18, seed=71)

    # split (time-aware, leakage-guarded) → split-manifest
    train, val, test, split_manifest = split_dataset(df, cfg, strategy="time")
    assert split_manifest.artifact == "split-manifest"
    assert split_manifest.leakage.time_ordered is True

    # train → model-card, and persist/reload the estimator
    est, card = train_model(train, cfg, model="logistic", seed=1)
    assert card.artifact == "model-card"
    assert card.train_metrics["auc"] > card.baseline_metrics["auc"]  # beats the floor
    model_path = tmp_path / "model.pkl"
    save_model(est, model_path)
    est = load_model(model_path)

    # compare the shortlist on stability
    ranked = compare_models(train, val, cfg, models=["logistic", "rf"])
    assert [r["model"] for r in ranked] and len(ranked) == 2

    # evaluate on held-out test → eval-report (with gain), reference for drift PSI
    ev = evaluate_model(est, test, cfg, reference_df=train)
    assert ev.artifact == "eval-report" and ev.gain and ev.segments

    # policy → policy-report
    pol = simulate_policy(est, test, cfg, save_rate=0.3, offer_cost=2.0)
    assert pol.artifact == "policy-report" and pol.n_targeted > 0
    assert pol.net_value == round(pol.expected_retained_value - pol.expected_spend, 2)

    # report → self-contained HTML from the artifacts
    html = build_html(ev.model_dump(), pol.model_dump(), card.model_dump())
    assert html.startswith("<!doctype html>")
    assert html.count("<figure>") == 4 and "data:image/png;base64," in html

    # monitor drift → drift-report (drift is present in the panel → retrain recommended)
    drift = monitor_drift(df, cfg)
    assert drift.artifact == "drift-report"
    assert drift.drifted and drift.retrain_recommended is True

    # Every persisted artifact carries lineage.
    for art in (split_manifest, card, ev, pol, drift):
        assert art.parent_sha256 and len(art.parent_sha256) == 64
