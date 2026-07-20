"""Report charts — one tested source of visuals (clean, light, validated palette).

Each function returns PNG bytes; ``report.py`` embeds them (base64) into a self-contained
HTML. matplotlib/Agg only — deterministic, static (the interactive Streamlit dashboard is a
separate slice). Colors are the validated clean-light palette; axes/grid are recessive, marks
are thin, single y-axis, legend only for ≥2 series, selective direct labels.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated clean-light palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"


def _ax(figsize: tuple[float, float] = (6.6, 4.0)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _dollars(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _pos: f"${v:,.0f}")


def policy_tradeoff_chart(curve: list[dict]) -> bytes:
    """Retained + net value vs. customers targeted — the retention trade-off (money chart)."""
    xs = [0] + [p["n_targeted"] for p in curve]
    retained = [0.0] + [p["retained_value"] for p in curve]
    net = [0.0] + [p["net"] for p in curve]

    fig, ax = _ax()
    ax.plot(xs, retained, color=MUTED, linewidth=2.0, label="Retained value")
    ax.plot(xs, net, color=BLUE, linewidth=2.5, label="Net value")

    peak = max(range(len(net)), key=lambda i: net[i])
    ax.scatter([xs[peak]], [net[peak]], color=BLUE, s=34, zorder=5)
    ax.annotate(
        f"peak net ${net[peak]:,.0f}",
        (xs[peak], net[peak]),
        textcoords="offset points",
        xytext=(8, -2),
        color=INK,
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("customers targeted", color=INK2, fontsize=10)
    _dollars(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    _title(ax, "Retention policy — value vs. customers targeted")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    return _png(fig)


def gain_chart(gain: list[dict]) -> bytes:
    """Cumulative churners captured vs. customers contacted (top-score first) + random baseline."""
    x = [0.0] + [g["cum_pop"] for g in gain]
    y = [0.0] + [g["cum_capture"] for g in gain]
    fig, ax = _ax()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.5, linestyle="--", label="random")
    ax.plot(x, y, color=BLUE, linewidth=2.5, label="model")
    ax.set_xlabel("fraction of customers contacted", color=INK2, fontsize=10)
    ax.set_ylabel("fraction of churners captured", color=INK2, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    _title(ax, "Gain — churners captured vs. customers contacted")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    return _png(fig)


def calibration_chart(calibration: list[dict]) -> bytes:
    """Predicted vs. observed churn rate per probability bin; on the diagonal = well-calibrated."""
    x = [c["mean_pred"] for c in calibration]
    y = [c["obs_rate"] for c in calibration]
    hi = max([*x, *y, 0.01]) * 1.08
    fig, ax = _ax()
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1.5, linestyle="--", label="perfect")
    ax.plot(x, y, color=BLUE, linewidth=2.0, marker="o", markersize=5, label="model")
    ax.set_xlabel("mean predicted probability", color=INK2, fontsize=10)
    ax.set_ylabel("observed churn rate", color=INK2, fontsize=10)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    _title(ax, "Calibration — predicted vs. observed")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    return _png(fig)


def segment_lift_chart(segment: dict, label: str = "segment") -> bytes:
    """Top-decile lift per segment level — single hue, direct-labeled bars, baseline at 1×."""
    levels = list(segment.keys())
    lifts = [float(segment[lv]["lift"]) for lv in levels]
    fig, ax = _ax(figsize=(6.6, 3.6))
    bars = ax.bar(levels, lifts, color=BLUE, width=0.6)
    for b, v in zip(bars, lifts):
        ax.annotate(
            f"{v:.2f}x",
            (b.get_x() + b.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_ylabel("top-decile lift", color=INK2, fontsize=10)
    ax.set_ylim(0, max(lifts) * 1.25 if lifts else 1)
    _title(ax, f"Top-decile lift by {label}")
    return _png(fig)


def qini_curve_chart(curve: list[dict]) -> bytes:
    """Cumulative incremental retentions vs. customers targeted by predicted uplift + random."""
    x = [p["frac"] for p in curve]
    q = [p["qini"] for p in curve]
    rand = [p["random"] for p in curve]
    fig, ax = _ax()
    ax.plot(x, rand, color=MUTED, linewidth=1.5, linestyle="--", label="random")
    ax.plot(x, q, color=BLUE, linewidth=2.5, label="uplift model")
    ax.set_xlabel("fraction of customers targeted (by predicted uplift)", color=INK2, fontsize=10)
    ax.set_ylabel("cumulative incremental retentions", color=INK2, fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:,.0f}")
    _title(ax, "Qini — incremental retentions vs. targeting")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    return _png(fig)


def uplift_vs_risk_chart(strategies: dict) -> bytes:
    """True net value from targeting by risk vs. by uplift at one budget (the v2 money chart)."""
    labels = ["target by risk", "target by uplift"]
    nets = [strategies["risk"]["true_net_value"], strategies["uplift"]["true_net_value"]]
    dogs = [
        strategies["risk"]["sleeping_dogs_treated"],
        strategies["uplift"]["sleeping_dogs_treated"],
    ]
    fig, ax = _ax(figsize=(6.6, 3.8))
    bars = ax.bar(labels, nets, color=[MUTED, BLUE], width=0.55)
    for b, v, sd in zip(bars, nets, dogs):
        ax.annotate(
            f"${v:,.0f}\n{sd:,} sleeping dogs",
            (b.get_x() + b.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )
    _dollars(ax)
    ax.set_ylabel("true net value", color=INK2, fontsize=10)
    ax.set_ylim(0, max(nets) * 1.25 if nets else 1)
    _title(ax, "Retention value — risk vs. uplift (equal budget)")
    return _png(fig)
