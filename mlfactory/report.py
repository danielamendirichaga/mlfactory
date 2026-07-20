"""HTML report — assembles a self-contained ``report.html`` from the typed artifacts.

Pure rendering step over the contracts: it reads the ``eval-report`` (and optional
``policy-report`` / ``model-card``) and lays out a clean-light page — headline stat tiles and
the charts from :mod:`mlfactory.charts` (embedded as base64, so the file is shareable with no
external assets). No compute here; every number comes from an artifact.
"""

from __future__ import annotations

import base64

from mlfactory import charts

_CSS = """
:root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--blue:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
.wrap{max-width:820px;margin:0 auto;padding:48px 24px 72px}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;font-weight:600;
color:var(--blue);margin-bottom:10px}
h1{font-size:30px;font-weight:700;margin:0 0 8px;text-wrap:balance;letter-spacing:-0.01em}
.meta{color:var(--ink2);font-size:14px;margin:0}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:32px 0}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:16px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-size:26px;font-weight:700;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.tile .s{font-size:12px;color:var(--ink2)}
figure{margin:16px 0 0;background:var(--surface);border:1px solid var(--grid);
border-radius:12px;padding:20px}
figure img{display:block;width:100%;height:auto}
figcaption{color:var(--ink2);font-size:13.5px;margin-top:14px;max-width:64ch}
.foot{color:var(--muted);font-size:12.5px;margin-top:32px}
h2.section{font-size:21px;font-weight:700;margin:44px 0 2px;letter-spacing:-0.01em}
p.lede{color:var(--ink2);font-size:14px;margin:0 0 4px}
table.dec{width:100%;border-collapse:collapse;margin-top:16px;font-size:13.5px}
table.dec th{text-align:left;color:var(--muted);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--grid)}
table.dec td{padding:6px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
table.dec td:first-child,table.dec th:first-child{color:var(--ink2)}
@media (max-width:640px){.tiles{grid-template-columns:repeat(2,1fr)}}
"""


def _tile(k: str, v: str, s: str) -> str:
    return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def _figure(png: bytes, caption: str) -> str:
    b64 = base64.b64encode(png).decode()
    return (
        f'<figure><img alt="{caption}" src="data:image/png;base64,{b64}"/>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _uplift_section(qini_report: dict | None, policy_contrast: dict | None) -> str:
    """The optional v2 block: Qini + the risk-vs-uplift contrast, from those artifacts."""
    if not qini_report and not policy_contrast:
        return ""
    tiles = []
    if qini_report:
        tiles.append(
            _tile("Qini coefficient", f"{qini_report['qini_coefficient']:,.0f}", "targeting power")
        )
        rec = qini_report.get("tau_recovery_corr")
        if rec is not None:
            tiles.append(_tile("τ recovery", f"{rec:+.2f}", "vs. ground truth"))
    if policy_contrast:
        tiles.append(
            _tile(
                "uplift advantage",
                f"${policy_contrast['uplift_net_advantage']:,.0f}",
                "net vs. risk",
            )
        )
        tiles.append(
            _tile(
                "sleeping dogs avoided",
                f"{policy_contrast['sleeping_dogs_avoided']:,}",
                "vs. risk targeting",
            )
        )

    figs = []
    if qini_report and qini_report.get("qini_curve"):
        figs.append(
            _figure(
                charts.qini_curve_chart(qini_report["qini_curve"]),
                "Qini — the extra retentions won as we target down the predicted-uplift list, "
                "versus random.",
            )
        )
    if policy_contrast:
        figs.append(
            _figure(
                charts.uplift_vs_risk_chart(policy_contrast["strategies"]),
                "At equal budget, targeting by uplift keeps more true value and treats far fewer "
                "sleeping dogs — scored on the counterfactual, not the model's own estimate.",
            )
        )

    dec = ""
    if qini_report and qini_report.get("uplift_deciles"):
        rows = "".join(
            f"<tr><td>{r['decile']}</td><td>{r['n']:,}</td>"
            f"<td>{'—' if r['obs_uplift'] is None else format(r['obs_uplift'], '+.4f')}</td></tr>"
            for r in qini_report["uplift_deciles"]
        )
        dec = (
            '<table class="dec"><thead><tr><th>decile (by predicted uplift)</th>'
            "<th>customers</th><th>observed uplift</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return (
        '<h2 class="section">Uplift — whom does the offer actually change?</h2>'
        '<p class="lede">Targeting by <em>risk</em> spends on lost causes and sleeping dogs; '
        "targeting by <em>uplift</em> finds the persuadables.</p>"
        f'<section class="tiles">{"".join(tiles)}</section>'
        f"{''.join(figs)}"
        f"{dec}"
    )


def build_html(
    eval_report: dict,
    policy_report: dict | None = None,
    model_card: dict | None = None,
    qini_report: dict | None = None,
    policy_contrast: dict | None = None,
) -> str:
    """Render a complete, self-contained HTML report document (v1 + optional v2 uplift section)."""
    mx = eval_report["metrics"]
    family = (model_card or {}).get("model_family", "model")
    n_rows = eval_report["n_rows"]

    tiles = [
        _tile("test AUC", f"{mx['auc']:.3f}", "discrimination"),
        _tile("top-decile lift", f"{mx['top_decile_lift']:.2f}&times;", "targeting"),
    ]
    if policy_report:
        roi = policy_report.get("roi")
        tiles += [
            _tile("net value", f"${policy_report['net_value']:,.0f}", "retention policy"),
            _tile("ROI", f"{roi:.2f}&times;" if roi else "n/a", "at full budget"),
        ]
    else:
        tiles += [
            _tile("KS", f"{mx['ks']:.3f}", "separation"),
            _tile("ECE", f"{mx['ece']:.3f}", "calibration error"),
        ]

    figs: list[str] = []
    if eval_report.get("gain"):
        figs.append(
            _figure(
                charts.gain_chart(eval_report["gain"]),
                "Gain — the share of churners captured as we contact more customers, top-scored first, "
                "versus the random baseline.",
            )
        )
    if eval_report.get("calibration"):
        figs.append(
            _figure(
                charts.calibration_chart(eval_report["calibration"]),
                "Calibration — mean predicted probability vs. observed churn rate; points on the diagonal "
                "mean the probabilities can be trusted as probabilities.",
            )
        )
    segments = eval_report.get("segments") or {}
    seg_col = "plan_tier" if "plan_tier" in segments else next(iter(segments), None)
    if seg_col:
        figs.append(
            _figure(
                charts.segment_lift_chart(segments[seg_col], seg_col),
                f"Top-decile lift by {seg_col} — where the model targets best (dashed line = no lift).",
            )
        )
    if policy_report:
        figs.append(
            _figure(
                charts.policy_tradeoff_chart(policy_report["tradeoff_curve"]),
                "Retention policy — net value climbs then plateaus as we target further down the ranked "
                "list; the gap to retained value is the offer spend.",
            )
        )

    uplift_section = _uplift_section(qini_report, policy_contrast)
    provenance = "eval-report" + (" + policy-report" if policy_report else "")
    if qini_report or policy_contrast:
        provenance += " + qini-report / policy-contrast"

    body = (
        '<div class="wrap">'
        '<header><div class="eyebrow">mlfactory · retention report</div>'
        "<h1>SaaS churn — model &amp; policy</h1>"
        f'<p class="meta">Held-out test: {n_rows:,} rows · model: {family}</p></header>'
        f'<section class="tiles">{"".join(tiles)}</section>'
        f"{''.join(figs)}"
        f"{uplift_section}"
        f'<p class="foot">Generated from typed artifacts ({provenance}). Synthetic data; no PII.</p>'
        "</div>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>mlfactory report</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )
