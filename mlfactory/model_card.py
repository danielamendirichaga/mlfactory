"""Render a markdown **model card** from the model + eval artifacts — the DS go/no-go surface.

Pure formatting over the typed artifacts (no compute): given a ``model-card`` artifact dict (from
``train``) and an optional ``eval-report`` dict (from ``evaluate``), produce an 8-section markdown
card (Purpose · Training Data · Features · Performance · Calibration · Slices · Limitations ·
Lineage). This is the human-reviewable deliverable a data scientist signs off on before shipping.
"""

from __future__ import annotations

from typing import Optional

SECTIONS = (
    "Purpose",
    "Training Data",
    "Features",
    "Performance",
    "Calibration",
    "Slices",
    "Limitations",
    "Lineage",
)


def _fmt(v: object, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def gen_model_card(
    model_card: dict,
    eval_report: Optional[dict] = None,
    *,
    target: str = "churn_next_30d",
) -> str:
    """Render the markdown model card. ``eval_report`` adds the held-out Performance/Calibration/Slices."""
    mc = model_card
    em = (eval_report or {}).get("metrics", {})
    fam = mc.get("model_family", "model")
    flags = [
        k for k in ("tuned", "smote", "calibrated", "early_stopping", "engineered") if mc.get(k)
    ]
    flag_s = f" (+{', '.join(flags)})" if flags else ""
    lines: list[str] = [f"# Model Card — {fam}", ""]

    # Purpose
    lines += [
        "## Purpose",
        "",
        f"A **{fam}**{flag_s} classifier predicting `{target}` — the probability an account churns "
        "in the next cycle. Higher score = higher churn risk.",
        "",
    ]

    # Training Data
    lines += [
        "## Training Data",
        "",
        f"- Input content hash (`parent_sha256`): `{mc.get('parent_sha256', 'n/a')}`",
        f"- Fit options: tuned={mc.get('tuned')}, smote={mc.get('smote')}, "
        f"calibrated={mc.get('calibrated')}, early_stopping={mc.get('early_stopping')}, "
        f"engineered={mc.get('engineered')}",
        "",
    ]

    # Features
    feats = mc.get("features", []) or []
    lines += [
        "## Features",
        "",
        f"{mc.get('n_features', len(feats))} features (leakage-screened):",
        "",
        ", ".join(f"`{f}`" for f in feats) if feats else "_none recorded_",
        "",
    ]
    if mc.get("hyperparams"):
        hp = ", ".join(f"{k}={v}" for k, v in mc["hyperparams"].items())
        lines += [f"**Hyperparameters:** {hp}", ""]

    # Performance
    tm = mc.get("train_metrics", {})
    bm = mc.get("baseline_metrics", {})
    lines += ["## Performance", "", "| metric | value |", "| --- | --- |"]
    lines += [f"| baseline floor AUC | {_fmt(bm.get('auc'))} |"]
    lines += [f"| train AUC | {_fmt(tm.get('auc'))} |"]
    for label, key in (
        ("test AUC", "auc"),
        ("test PR-AUC", "pr_auc"),
        ("test KS", "ks"),
        ("top-decile lift", "top_decile_lift"),
        ("log-loss", "log_loss"),
    ):
        if key in em:
            lines += [f"| {label} | {_fmt(em[key])} |"]
    lines += [""]

    # Calibration
    if "ece" in em:
        lines += [
            "## Calibration",
            "",
            f"- Expected Calibration Error (ECE): **{_fmt(em['ece'])}**",
            "",
        ]

    # Slices
    segs = (eval_report or {}).get("segments") or {}
    if segs:
        lines += ["## Slices", ""]
        for dim, seg in segs.items():
            lines += [
                f"**by {dim}:**",
                "",
                "| segment | n | churn rate | AUC | lift |",
                "| --- | --- | --- | --- | --- |",
            ]
            for level, s in seg.items():
                lines += [
                    f"| {level} | {s.get('n')} | {_fmt(s.get('churn_rate'))} | "
                    f"{_fmt(s.get('auc'))} | {_fmt(s.get('lift'), 2)} |"
                ]
            lines += [""]

    # Limitations
    lines += [
        "## Limitations",
        "",
        "- Trained/evaluated on the **synthetic B2B SaaS reference domain** — not real production "
        "performance.",
        "- The score is a risk ranking, not a calibrated business decision; pair it with the policy layer.",
        "- A point-in-time snapshot; monitor for drift before relying on it next quarter.",
    ]
    for caveat in mc.get("caveats") or []:
        lines += [f"- {caveat}"]
    lines += [""]

    # Lineage
    lines += [
        "## Lineage",
        "",
        f"- artifact: `{mc.get('artifact', 'model-card')}` v{mc.get('version', '1.0')}",
        f"- input content hash (`parent_sha256`): `{mc.get('parent_sha256', 'n/a')}`",
    ]
    if eval_report is not None:
        lines += [f"- evaluated on {eval_report.get('n_rows', 'n/a')} held-out rows"]
    lines += [""]

    return "\n".join(lines).rstrip() + "\n"
