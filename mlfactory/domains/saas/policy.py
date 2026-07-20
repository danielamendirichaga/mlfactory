"""Retention policy simulation — cost-based, budget-constrained "who do we save?".

Risk scores become a *decision*. For each customer, the expected net value of making a
save-offer is::

    benefit(x) = save_rate · P(churn | x) · CLTV(x)  −  offer_cost

We target the **positive-benefit** customers, best-first, until the budget runs out (a count of
offers or a dollar cap), and report retained value, spend, net, ROI, a trade-off curve, and a
by-segment breakdown. This is per-customer cost-sensitive targeting: targeting only where
``benefit(x) > 0`` is a threshold on ``P(churn)`` that scales with each customer's CLTV.

``save_rate`` is a fixed assumption for the risk policy; :func:`contrast_policies` (v2) drops it,
replacing ``save_rate·P(churn)`` with a per-customer **uplift** estimate ``τ̂(x)`` and scoring both
strategies on the *true* counterfactual — the honest test of whether targeting persuadables beats
targeting the highest-risk. Reuses the fitted models + config; no new metric math.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from mlfactory.artifacts import ArtifactBase, content_hash
from mlfactory.config import ChurnConfig
from mlfactory.compute.model import feature_columns


class PolicyError(ValueError):
    """Raised when a policy cannot be simulated (e.g. no CLTV column)."""


class PolicyReport(ArtifactBase):
    artifact: Literal["policy-report"] = "policy-report"
    save_rate: float
    offer_cost: float
    budget: Optional[float] = None
    n_offers: Optional[int] = None
    n_customers: int
    n_eligible: int  # customers with benefit(x) > 0
    n_targeted: int
    expected_retained_value: float
    expected_spend: float
    net_value: float
    roi: Optional[float]
    tradeoff_curve: list[dict]
    segments: dict


def _target_set(
    benefit: np.ndarray, offer_cost: float, budget: Optional[float], n_offers: Optional[int]
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Rank by benefit desc; return ``(order, n_eligible, cap, mask)`` for the positive-benefit
    set kept within the budget — a count of offers or a dollar cap. Shared by both policies."""
    order = np.argsort(-benefit, kind="mergesort")  # best-first, stable
    n_eligible = int((benefit > 0).sum())
    cap = n_eligible
    if n_offers is not None:
        cap = min(cap, n_offers)
    if budget is not None:
        cap = min(cap, int(budget // offer_cost) if offer_cost > 0 else cap)
    mask = np.zeros(len(benefit), dtype=bool)
    mask[order[:cap]] = True
    return order, n_eligible, cap, mask


def simulate_policy(
    estimator,
    data: pd.DataFrame,
    config: ChurnConfig,
    save_rate: float = 0.3,
    offer_cost: float = 5.0,
    budget: Optional[float] = None,
    n_offers: Optional[int] = None,
    segment_col: str = "plan_tier",
) -> PolicyReport:
    """Score ``data``, rank by ``benefit(x)``, target under the budget, and report the outcome."""
    if budget is not None and n_offers is not None:
        raise PolicyError("give a budget ($) OR n_offers, not both")
    cols = config.columns
    if cols.value_col is None or cols.value_col not in data.columns:
        raise PolicyError("simulate-policy requires a value_col (CLTV) present in the data")

    data = data.reset_index(drop=True)
    numeric, categorical = feature_columns(data, config)
    p_churn = estimator.predict_proba(data[numeric + categorical])[:, 1]
    cltv = data[cols.value_col].to_numpy(dtype=float)

    gross = save_rate * p_churn * cltv  # expected value saved if we target them
    benefit = gross - offer_cost
    order, n_eligible, cap, mask = _target_set(benefit, offer_cost, budget, n_offers)

    retained = float(gross[mask].sum())
    spend = float(cap * offer_cost)
    net = round(retained - spend, 2)
    roi = round(retained / spend, 3) if spend > 0 else None

    # Trade-off curve: net value as we target more of the eligible list.
    cum_gross = np.cumsum(gross[order])
    points = sorted({int(round(f * n_eligible)) for f in (0.1, 0.25, 0.5, 0.75, 1.0)} | {cap})
    curve = [
        {
            "n_targeted": k,
            "spend": round(k * offer_cost, 2),
            "retained_value": round(float(cum_gross[k - 1]), 2),
            "net": round(float(cum_gross[k - 1] - k * offer_cost), 2),
        }
        for k in points
        if 0 < k <= n_eligible
    ]

    # By-segment breakdown of the targeted set.
    segments: dict = {}
    if segment_col in data.columns:
        for level in data[segment_col].dropna().unique():
            m = mask & (data[segment_col] == level).to_numpy()
            segments[str(level)] = {
                "n_targeted": int(m.sum()),
                "retained_value": round(float(gross[m].sum()), 2),
            }

    return PolicyReport(
        save_rate=save_rate,
        offer_cost=offer_cost,
        budget=budget,
        n_offers=n_offers,
        n_customers=len(data),
        n_eligible=n_eligible,
        n_targeted=cap,
        expected_retained_value=round(retained, 2),
        expected_spend=round(spend, 2),
        net_value=net,
        roi=roi,
        tradeoff_curve=curve,
        segments=segments,
        parent_sha256=content_hash(data),
    )


class PolicyContrast(ArtifactBase):
    artifact: Literal["policy-contrast"] = "policy-contrast"
    offer_cost: float
    save_rate: float
    budget: Optional[float] = None
    n_offers: Optional[int] = None
    n_customers: int
    strategies: dict  # {"risk": {...}, "uplift": {...}}
    uplift_net_advantage: float  # true net(uplift) − true net(risk)
    sleeping_dogs_avoided: int


def contrast_policies(
    risk_estimator: Any,
    uplift_model: Any,
    data: pd.DataFrame,
    config: ChurnConfig,
    offer_cost: float = 5.0,
    budget: Optional[float] = None,
    n_offers: Optional[int] = None,
    save_rate: float = 0.3,
) -> PolicyContrast:
    """Target by **risk** vs by **uplift** under one budget, scored on the true counterfactual.

    Risk ranks by ``save_rate·P̂(churn)·CLTV``; uplift ranks by ``τ̂·CLTV``. Each targeted set is
    then valued with the *true* effect ``true_uplift·CLTV`` (synthetic ground truth), so treating a
    sleeping dog costs value. Requires an A/B panel with ``true_uplift`` and a CLTV column.
    """
    cols = config.columns
    if cols.value_col is None or cols.value_col not in data.columns:
        raise PolicyError("policy contrast requires a value_col (CLTV) in the data")
    if "true_uplift" not in data.columns:
        raise PolicyError(
            "policy contrast needs `true_uplift` for honest scoring — use an A/B panel "
            "(`generate --treatment`)"
        )

    data = data.reset_index(drop=True)
    numeric, categorical = feature_columns(data, config)
    cltv = data[cols.value_col].to_numpy(dtype=float)
    true_tau = data["true_uplift"].to_numpy(dtype=float)

    p_churn = risk_estimator.predict_proba(data[numeric + categorical])[:, 1]
    tau_hat = uplift_model.predict_uplift(data)
    benefits = {
        "risk": save_rate * p_churn * cltv - offer_cost,
        "uplift": tau_hat * cltv - offer_cost,
    }

    strategies: dict = {}
    for name, benefit in benefits.items():
        _, _, cap, mask = _target_set(benefit, offer_cost, budget, n_offers)
        true_retained = float((true_tau[mask] * cltv[mask]).sum())  # honest: the causal value
        spend = float(cap * offer_cost)
        strategies[name] = {
            "n_targeted": cap,
            "true_retained_value": round(true_retained, 2),
            "spend": round(spend, 2),
            "true_net_value": round(true_retained - spend, 2),
            "roi": round(true_retained / spend, 3) if spend > 0 else None,
            "sleeping_dogs_treated": int(((true_tau < 0) & mask).sum()),
        }

    return PolicyContrast(
        offer_cost=offer_cost,
        save_rate=save_rate,
        budget=budget,
        n_offers=n_offers,
        n_customers=len(data),
        strategies=strategies,
        uplift_net_advantage=round(
            strategies["uplift"]["true_net_value"] - strategies["risk"]["true_net_value"], 2
        ),
        sleeping_dogs_avoided=(
            strategies["risk"]["sleeping_dogs_treated"]
            - strategies["uplift"]["sleeping_dogs_treated"]
        ),
        parent_sha256=content_hash(data),
    )
