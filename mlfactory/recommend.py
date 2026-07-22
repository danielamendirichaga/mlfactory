"""Recommendation engine — the copilot's judgment, as tested *rules* (not an LLM).

Each function reads a diagnostic (profile records, compare results, a drift report, an eval
report, or the config) and returns a :class:`Recommendation`: *what* mlfactory would advise,
*why*, and a structured *action*. The ``advise`` command prints these; the interactive ``run``
command acts on them at each checkpoint (you approve or override). The rules are deterministic
and unit-tested — judgment-*support* that stays in the tested core, so the final call is still a
human's, never the tool's.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import TREATMENT_COL
from mlfactory.compute.profile import high_corr_features


class Recommendation(BaseModel):
    gate: str  # features | split | model | policy | retrain | ship
    recommendation: str  # what to do, in one line
    rationale: str  # why
    action: dict  # structured suggestion the `run` command can act on


def recommend_features(profile_records: list[dict], leak_threshold: float = 0.5) -> Recommendation:
    """Flag likely-leakage features (extreme target correlation) for exclusion."""
    leaky = high_corr_features(profile_records, threshold=leak_threshold)
    if leaky:
        names = [c for c, _ in leaky]
        hits = ", ".join(f"{c} ({v:+.2f})" for c, v in leaky)
        return Recommendation(
            gate="features",
            recommendation=f"Exclude {', '.join(names)}",
            rationale=f"target correlation ≥ {leak_threshold} — likely leakage: {hits}",
            action={"exclude": names},
        )
    return Recommendation(
        gate="features",
        recommendation="Keep all features",
        rationale=f"no feature exceeds the {leak_threshold} leakage threshold",
        action={"exclude": []},
    )


def recommend_split(config: ChurnConfig) -> Recommendation:
    """Time-aware split when there's a date column; random for snapshot data."""
    if config.columns.date_col is not None:
        return Recommendation(
            gate="split",
            recommendation="Use a time-aware split",
            rationale="a date column is present → an out-of-time split is the honest test for churn",
            action={"strategy": "time"},
        )
    return Recommendation(
        gate="split",
        recommendation="Use a random split",
        rationale="no date column (snapshot data) → a time-aware split isn't possible",
        action={"strategy": "random"},
    )


def recommend_model(compare_rows: list[dict]) -> Recommendation:
    """Prefer the most *stable* model, ranked on the DS's chosen primary metric (not just peak AUC)."""
    if not compare_rows:
        return Recommendation(
            gate="model",
            recommendation="No models to compare",
            rationale="run `compare` on a shortlist first",
            action={"model": None},
        )

    def _score(r: dict) -> float:
        return float(r.get("primary", r["holdout_auc"]))

    metric = compare_rows[0].get("primary_metric", "auc")
    stable = [r for r in compare_rows if r.get("stable")]
    pool = stable or compare_rows
    pick = max(pool, key=_score)
    top = max(compare_rows, key=_score)
    why = f"{metric} {_score(pick):.3f}, stable (train→holdout AUC drop {pick['auc_drop']:+.3f})"
    if top["model"] != pick["model"]:
        why += (
            f"; {top['model']} scores higher ({_score(top):.3f}) but overfits "
            f"(drop {top['auc_drop']:+.3f}) — select on stability, not peak {metric}"
        )
    return Recommendation(
        gate="model",
        recommendation=f"Ship {pick['model']}",
        rationale=why,
        action={"model": pick["model"]},
    )


def recommend_policy(config: ChurnConfig) -> Recommendation:
    """Cost-based targeting when a value column is set; otherwise ask for one."""
    vc = config.columns.value_col
    if vc is not None:
        return Recommendation(
            gate="policy",
            recommendation="Target the positive-benefit customers under your budget",
            rationale=f"value column '{vc}' present → cost-based (save-value vs. offer-cost) targeting",
            action={"value_col": vc},
        )
    return Recommendation(
        gate="policy",
        recommendation="Set a value column to enable the policy",
        rationale="no value_col → can't weigh a customer's save-value against the offer cost",
        action={"value_col": None},
    )


def recommend_retrain(drift: dict) -> Recommendation:
    """Retrain when features have drifted past the threshold (never automatically)."""
    if drift.get("skipped"):
        return Recommendation(
            gate="retrain",
            recommendation="Drift check unavailable",
            rationale="snapshot data (no date column) → can't compare cohorts",
            action={"retrain": False},
        )
    if drift.get("retrain_recommended"):
        drifted = drift.get("drifted", [])
        shown = ", ".join(drifted[:5]) + (" …" if len(drifted) > 5 else "")
        return Recommendation(
            gate="retrain",
            recommendation="Retrain the model",
            rationale=f"{len(drifted)} feature(s) drifted past the threshold: {shown}",
            action={"retrain": True},
        )
    return Recommendation(
        gate="retrain",
        recommendation="No retrain needed",
        rationale="no feature drifted past the threshold",
        action={"retrain": False},
    )


def recommend_ship(
    eval_report: dict, min_auc: float = 0.65, max_ece: float = 0.10
) -> Recommendation:
    """A go/no-go read on a held-out eval — discrimination and calibration must both clear the bar."""
    m = eval_report["metrics"]
    auc, ece = float(m["auc"]), float(m["ece"])
    ok = auc >= min_auc and ece <= max_ece
    why = (
        f"AUC {auc:.3f} ({'≥' if auc >= min_auc else '<'} {min_auc}), "
        f"ECE {ece:.3f} ({'≤' if ece <= max_ece else '>'} {max_ece})"
    )
    return Recommendation(
        gate="ship",
        recommendation="Ship it" if ok else "Not ready — investigate before shipping",
        rationale=why,
        action={"ship": ok},
    )


def recommend_experiment(df: pd.DataFrame) -> Recommendation:
    """Is there a randomized experiment in the data → is uplift (v2) even available?"""
    if TREATMENT_COL in df.columns and int(df[TREATMENT_COL].nunique(dropna=True)) >= 2:
        rate = float((df[TREATMENT_COL] == 1).mean())
        return Recommendation(
            gate="uplift",
            recommendation="Uplift (v2) is available",
            rationale=(
                f"a randomized '{TREATMENT_COL}' column is present ({rate:.0%} treated) — "
                "you can target by causal effect (`train-uplift`), not just risk"
            ),
            action={"uplift": True},
        )
    return Recommendation(
        gate="uplift",
        recommendation="v1 (risk) pipeline — uplift (v2) not available",
        rationale=(
            f"no '{TREATMENT_COL}' column → no experiment; uplift needs a randomized A/B test "
            "(treatment vs. control), which observational churn data doesn't have"
        ),
        action={"uplift": False},
    )
