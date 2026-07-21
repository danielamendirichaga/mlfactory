"""mlfactory command-line interface.

v1 pipeline: init -> generate -> validate -> profile -> metrics -> split -> train ->
compare -> evaluate -> simulate-policy -> report -> monitor.
v2 (uplift/causal): generate --treatment -> train-uplift -> (uplift-eval, uplift policy).
Copilot: advise (pre-flight recommendations) and run (interactive, checkpointed pipeline).
Factory: engineer-features, leakage-scan, validate-artifact, export-schemas, gen-model-card
(most stage commands take --json). The agent layer lives in .claude/ (see .claude/README.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mlfactory import __version__
from mlfactory.config import CONFIG_TEMPLATE

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="mlfactory — an LLM-orchestrated ML factory: a deterministic, tested CLI for the "
    "data->model pipeline, with a B2B SaaS churn reference domain.",
)


@app.callback()
def main() -> None:
    """mlfactory — an LLM-orchestrated ML factory (deterministic tested CLI + typed lineage
    artifacts), with a B2B SaaS churn reference domain.

    A callback is defined so Typer keeps subcommand mode even while only one
    command exists; without it, a single-command Typer app treats the command
    name as a stray argument.
    """


@app.command()
def version() -> None:
    """Print the installed mlfactory version."""
    typer.echo(f"mlfactory {__version__}")


@app.command()
def init(
    path: Path = typer.Option(
        Path("churn.yaml"), "--path", help="Where to write the config template."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite the file if it already exists."),
) -> None:
    """Scaffold a churn.yaml config template to point mlfactory at your data."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists — use --force to overwrite.")
        raise typer.Exit(code=1)
    path.write_text(CONFIG_TEMPLATE)
    typer.echo(f"Wrote {path}. Edit it to point mlfactory at your data.")


@app.command()
def generate(
    out: Path = typer.Option(
        Path("data/churn_panel.parquet"), "--out", help="Output parquet path."
    ),
    accounts: int = typer.Option(8000, "--accounts", help="Number of accounts."),
    months: int = typer.Option(24, "--months", help="Number of monthly cohorts."),
    seed: int = typer.Option(42, "--seed", help="RNG seed (deterministic)."),
    treatment: bool = typer.Option(
        False, "--treatment", help="Overlay a randomized A/B test (adds uplift columns, for v2)."
    ),
) -> None:
    """Generate the deterministic synthetic SaaS churn panel (no real data)."""
    from mlfactory.domains.saas.generate import make_panel, summarize

    df = make_panel(n_accounts=accounts, n_months=months, seed=seed, treatment=treatment)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    typer.echo(f"Wrote {out}")
    typer.echo(summarize(df))


def _load(config_path: Path):
    """Load config + data for a command, exiting cleanly (no traceback) on failure."""
    from mlfactory.config import ConfigError, load_config
    from mlfactory.source import SourceError, load_data

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    try:
        df = load_data(cfg)
    except SourceError as exc:
        typer.echo(f"Could not load data: {exc}")
        raise typer.Exit(code=1) from exc
    return cfg, df


@app.command()
def validate(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
) -> None:
    """Check that the configured dataset is usable by mlfactory (fails gracefully)."""
    from mlfactory.validate import validate as run_validate

    cfg, df = _load(config)
    report = run_validate(df, cfg)
    typer.echo(report.render())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def profile(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
) -> None:
    """Profile every column of the configured dataset (EDA numbers)."""
    import pandas as pd

    from mlfactory.compute.profile import high_corr_features, profile_frame

    cfg, df = _load(config)
    records = profile_frame(df, cfg)
    table = pd.DataFrame(records)
    order = [
        "column",
        "role",
        "null_rate",
        "n_unique",
        "target_corr",
        "mean",
        "std",
        "min",
        "q25",
        "q50",
        "q75",
        "max",
    ]
    table = table[[c for c in order if c in table.columns]]

    typer.echo(
        f"Profile of {len(df):,} rows × {df.shape[1]} columns  (target: {cfg.columns.target_col})"
    )
    typer.echo("")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        typer.echo(table.to_string(index=False, na_rep=""))

    leaky = high_corr_features(records, threshold=0.5)
    if leaky:
        typer.echo("")
        hits = ", ".join(f"{c} ({v:+.2f})" for c, v in leaky)
        typer.echo(f"⚠ high target correlation — possible leakage: {hits}")


@app.command()
def metrics(
    score_col: str = typer.Option(..., "--score-col", help="Column to treat as the risk score."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    reference: Path = typer.Option(
        None, "--reference", help="Optional reference parquet for score PSI (drift)."
    ),
    n_bins: int = typer.Option(10, "--n-bins", help="Number of quantile deciles."),
) -> None:
    """Report discrimination/targeting metrics for a score column vs. the target."""
    import pandas as pd

    from mlfactory.compute import metrics as m

    cfg, df = _load(config)
    label = cfg.columns.target_col
    if score_col not in df.columns:
        typer.echo(f"score column {score_col!r} not found in data.")
        raise typer.Exit(code=1)
    y = (df[label] == cfg.columns.positive_value).astype(int)
    s = pd.to_numeric(df[score_col], errors="coerce")

    ks = m.ks_table(y, s, n_bins=n_bins)
    typer.echo(f"Metrics for score '{score_col}' vs target '{label}'  ({len(df):,} rows)")
    typer.echo("")
    typer.echo(f"  ROC-AUC          : {m.roc_auc(y, s):.4f}")
    typer.echo(f"  PR-AUC (AP)      : {m.average_precision(y, s):.4f}")
    typer.echo(f"  KS (decile)      : {ks.ks:.4f}   over {ks.n_bins} deciles")
    typer.echo(f"  top-decile lift  : {m.top_decile_lift(y, s):.3f}x")
    typer.echo(f"  rank-order breaks: {m.rank_order_breaks(y, s, n_bins=n_bins)}   (0 = clean)")

    if reference is not None:
        ref = pd.read_parquet(reference)
        if score_col not in ref.columns:
            typer.echo(f"\n⚠ score column {score_col!r} not in reference — skipping PSI.")
        else:
            val = m.psi(ref[score_col], df[score_col], n_bins=n_bins)
            typer.echo(
                f"\n  score PSI (ref→data): {val:.4f}   (<0.1 stable, 0.1–0.25 moderate, >0.25 major)"
            )


@app.command()
def split(
    strategy: str = typer.Option("time", "--strategy", help="time | grouped | random."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    out_dir: Path = typer.Option(
        Path("data/splits"), "--out-dir", help="Where to write the splits + manifest."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed (grouped/random)."),
) -> None:
    """Split into train/val/test with a leakage guard; writes parquets + a split-manifest."""
    from mlfactory.compute.split import SplitError, split_dataset

    cfg, df = _load(config)
    try:
        train, val, test, manifest = split_dataset(df, cfg, strategy=strategy, seed=seed)
    except SplitError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    manifest.write_json(out_dir / "split-manifest.json")

    typer.echo(f"Split ({strategy}) → {out_dir}")
    for name, info in (("train", manifest.train), ("val", manifest.val), ("test", manifest.test)):
        typer.echo(f"  {name}: {info.rows:,} rows, churn {info.positive_rate:.1%}")
    lk = manifest.leakage
    if lk.status == "warn":
        typer.echo(
            f"  ⚠ entity leakage: {lk.account_overlap:,} accounts in BOTH train & test "
            "— use --strategy time"
        )
    else:
        detail = "expected for time split" if strategy == "time" else "accounts disjoint"
        typer.echo(f"  ✔ leakage guard ok ({lk.account_overlap:,} account overlap — {detail})")


@app.command()
def train(
    train: Path = typer.Option(..., "--train", help="Training parquet (a split output)."),
    model: str = typer.Option("logistic", "--model", help="logistic | tree | rf | xgboost."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    model_out: Path = typer.Option(
        Path("data/model.pkl"), "--model-out", help="Where to persist the fitted model."
    ),
    smote: bool = typer.Option(False, "--smote", help="Oversample the minority class (SMOTE)."),
    calibrate: bool = typer.Option(False, "--calibrate", help="Isotonic probability calibration."),
    tune: bool = typer.Option(False, "--tune", help="Run the hyperparameter search."),
    early_stopping: bool = typer.Option(
        False, "--early-stopping", help="XGBoost early stopping (mode-aware inner-val)."
    ),
    optuna: bool = typer.Option(
        False, "--optuna", help="Optuna TPE hyperparameter search (seeded)."
    ),
    trials: int = typer.Option(30, "--trials", help="Optuna trials (used with --optuna)."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    json_out: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON summary."),
) -> None:
    """Fit a model from the menu (leakage-safe) and report it against the baseline floor."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.compute.model import ModelError, feature_columns, save_model, train_model
    from mlfactory.compute.profile import high_corr_features, profile_frame

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    df = pd.read_parquet(train)

    # Safety: warn if a feature about to be used looks like leakage (extreme target corr).
    numeric, categorical = feature_columns(df, cfg)
    used = set(numeric) | set(categorical)
    leaky = [
        (c, v) for c, v in high_corr_features(profile_frame(df, cfg), threshold=0.6) if c in used
    ]
    if leaky and not json_out:
        hits = ", ".join(f"{c} (|corr|={abs(v):.2f})" for c, v in leaky)
        typer.echo(f"⚠ possible leakage in features: {hits} — consider excluding it.\n")

    try:
        estimator, card = train_model(
            df,
            cfg,
            model=model,
            smote=smote,
            calibrate=calibrate,
            tune=tune,
            early_stopping=early_stopping,
            optuna=optuna,
            n_trials=trials,
            seed=seed,
        )
    except ModelError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_model(estimator, model_out)
    card.write_json(model_out.with_suffix(".card.json"))

    tags = "".join(
        t
        for t, on in (
            (" +smote", smote),
            (" +calibrated", calibrate),
            (" +tuned", tune),
            (" +early-stop", early_stopping),
            (" +optuna", optuna),
        )
        if on
    )
    tm, bm = card.train_metrics, card.baseline_metrics
    if json_out:
        import json

        typer.echo(
            json.dumps(
                {
                    "command": "train",
                    "model": model,
                    "model_out": str(model_out),
                    "n_features": card.n_features,
                    "train_metrics": tm,
                    "baseline_metrics": bm,
                }
            )
        )
        return
    typer.echo(
        f"Trained {model}{tags} on {len(df):,} rows ({card.n_features} features) → {model_out}"
    )
    typer.echo(
        f"  train : AUC {tm['auc']:.4f} | KS {tm['ks']:.4f} | top-decile lift {tm['top_decile_lift']:.2f}x"
    )
    typer.echo(f"  floor : AUC {bm['auc']:.4f}  (majority-class baseline to beat)")


@app.command()
def compare(
    train: Path = typer.Option(..., "--train", help="Training parquet (a split output)."),
    holdout: Path = typer.Option(
        ..., "--holdout", help="Held-out parquet to rank on (e.g. the val split)."
    ),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    models: str = typer.Option("", "--models", help="Comma-separated subset (default: all)."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
) -> None:
    """Fit the model shortlist and rank on held-out performance AND stability."""
    import pandas as pd

    from mlfactory.compute.compare import compare_models
    from mlfactory.config import ConfigError, load_config
    from mlfactory.compute.model import MODELS, ModelError

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    tr, ho = pd.read_parquet(train), pd.read_parquet(holdout)
    shortlist = [x.strip() for x in models.split(",") if x.strip()] or list(MODELS)
    try:
        rows = compare_models(tr, ho, cfg, models=shortlist, seed=seed)
    except ModelError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    table = pd.DataFrame(rows)
    for col in ("holdout_auc", "holdout_ks", "holdout_pr_auc"):
        table[col] = table[col].round(4)
    table["holdout_lift"] = table["holdout_lift"].round(2)
    table["stable"] = table["stable"].map({True: "✔", False: ""})

    typer.echo(f"Model comparison  ({len(tr):,} train → {len(ho):,} holdout rows)")
    typer.echo("")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        typer.echo(table.to_string(index=False))

    best = rows[0]
    stable = [r for r in rows if r["stable"]]
    typer.echo("")
    typer.echo(f"  best held-out AUC : {best['model']} ({best['holdout_auc']:.4f})")
    if stable:
        pick = min(stable, key=lambda r: r["auc_drop"])
        typer.echo(
            f"  most stable       : {pick['model']} (auc_drop {pick['auc_drop']:+.4f}, "
            f"score-PSI {pick['score_psi']:.3f}) — prefer for a model you'll trust next quarter"
        )
    else:
        typer.echo("  ⚠ none passed the stability gate (auc_drop < 0.05 and score-PSI < 0.2)")


@app.command()
def evaluate(
    model: Path = typer.Option(..., "--model", help="Persisted fitted model (.pkl)."),
    test: Path = typer.Option(..., "--test", help="Held-out parquet to score."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    reference: Path = typer.Option(
        None, "--reference", help="Optional reference parquet for score-PSI drift."
    ),
    threshold: float = typer.Option(0.5, "--threshold", help="Cutoff for precision/recall/F1."),
    report_out: Path = typer.Option(
        Path("data/eval-report.json"), "--report-out", help="Where to write the eval-report."
    ),
) -> None:
    """Evaluate a saved model on held-out data — the full metric pack + per-segment + drift."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.compute.evaluate import evaluate_model
    from mlfactory.compute.model import load_model

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    est = load_model(model)
    test_df = pd.read_parquet(test)
    ref_df = pd.read_parquet(reference) if reference is not None else None
    try:
        report = evaluate_model(est, test_df, cfg, reference_df=ref_df, threshold=threshold)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(report_out)

    mx = report.metrics
    typer.echo(f"Held-out evaluation  ({report.n_rows:,} rows) → {report_out}")
    typer.echo("")
    typer.echo(
        f"  AUC {mx['auc']:.4f} | PR-AUC {mx['pr_auc']:.4f} | KS {mx['ks']:.4f} | "
        f"top-decile lift {mx['top_decile_lift']:.2f}x | rank-order breaks {mx['rank_order_breaks']}"
    )
    typer.echo(
        f"  @{threshold:g}: precision {mx['precision']:.3f} | recall {mx['recall']:.3f} | "
        f"F1 {mx['f1']:.3f} | log-loss {mx['log_loss']:.4f} | ECE {mx['ece']:.4f}"
    )
    if report.score_psi is not None:
        typer.echo(
            f"  score-PSI (reference→test): {report.score_psi:.4f}   (<0.1 stable, >0.25 major)"
        )

    for col, seg in report.segments.items():
        typer.echo(f"\n  by {col}:")
        for level, s in seg.items():
            auc = f"{s['auc']:.3f}" if s["auc"] is not None else "  n/a"
            typer.echo(
                f"    {level:<12} n={s['n']:>6,}  churn {s['churn_rate']:.1%}  AUC {auc}  lift {s['lift']:.2f}x"
            )


@app.command("simulate-policy")
def simulate_policy_cmd(
    model: Path = typer.Option(..., "--model", help="Persisted fitted model (.pkl)."),
    data: Path = typer.Option(..., "--data", help="Customer parquet to target."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    save_rate: float = typer.Option(
        0.3, "--save-rate", help="P(offer rescues a would-be churner)."
    ),
    offer_cost: float = typer.Option(5.0, "--offer-cost", help="Cost of one save-offer ($)."),
    budget: Optional[float] = typer.Option(None, "--budget", help="Total offer budget ($)."),
    n_offers: Optional[int] = typer.Option(None, "--n-offers", help="Max number of offers."),
    report_out: Path = typer.Option(
        Path("data/policy-report.json"), "--report-out", help="Where to write the policy-report."
    ),
) -> None:
    """Cost-based retention targeting: whom to save under a budget, and the ROI."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.compute.model import load_model
    from mlfactory.domains.saas.policy import PolicyError, simulate_policy

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    est = load_model(model)
    df = pd.read_parquet(data)
    try:
        report = simulate_policy(
            est,
            df,
            cfg,
            save_rate=save_rate,
            offer_cost=offer_cost,
            budget=budget,
            n_offers=n_offers,
        )
    except PolicyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(report_out)

    if budget is not None:
        limit = f"${budget:,.0f} budget"
    elif n_offers is not None:
        limit = f"{n_offers:,} offers"
    else:
        limit = "unlimited budget"
    typer.echo(
        f"Retention policy  (save_rate {save_rate:g}, offer ${offer_cost:g}, {limit})  → {report_out}"
    )
    typer.echo("")
    typer.echo(
        f"  target {report.n_targeted:,} of {report.n_eligible:,} profitable "
        f"({report.n_customers:,} customers)"
    )
    roi = f" | ROI {report.roi:.2f}x" if report.roi is not None else ""
    typer.echo(
        f"  retained value ${report.expected_retained_value:,.0f} | "
        f"spend ${report.expected_spend:,.0f} | net ${report.net_value:,.0f}{roi}"
    )
    if report.segments:
        typer.echo("\n  targeted by plan_tier:")
        for level, s in report.segments.items():
            typer.echo(
                f"    {level:<12} {s['n_targeted']:>6,} offers → ${s['retained_value']:,.0f} retained"
            )
    typer.echo(
        f"\n  (save_rate {save_rate:g} is a fixed v1 assumption; uplift modeling replaces it in v2)"
    )


@app.command()
def report(
    eval_report: Path = typer.Option(..., "--eval", help="eval-report.json (from `evaluate`)."),
    policy: Path = typer.Option(
        None, "--policy", help="policy-report.json (from `simulate-policy`)."
    ),
    model_card: Path = typer.Option(None, "--model-card", help="model-card JSON (from `train`)."),
    qini: Path = typer.Option(None, "--qini", help="qini-report.json (from `uplift-eval`, v2)."),
    contrast: Path = typer.Option(
        None, "--contrast", help="policy-contrast.json (from `policy-contrast`, v2)."
    ),
    out: Path = typer.Option(
        Path("data/report.html"), "--out", help="Where to write the HTML report."
    ),
) -> None:
    """Render a shareable, self-contained HTML report from the pipeline artifacts."""
    import json

    from mlfactory.report import build_html

    ev = json.loads(eval_report.read_text())
    pol = json.loads(policy.read_text()) if policy is not None else None
    mc = json.loads(model_card.read_text()) if model_card is not None else None
    qr = json.loads(qini.read_text()) if qini is not None else None
    pc = json.loads(contrast.read_text()) if contrast is not None else None

    html = build_html(ev, pol, mc, qini_report=qr, policy_contrast=pc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    typer.echo(f"Wrote {out}  ({len(html):,} bytes) — open it in a browser.")


@app.command()
def monitor(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    threshold: float = typer.Option(0.25, "--threshold", help="PSI drift threshold for retrain."),
    report_out: Path = typer.Option(
        Path("data/drift-report.json"), "--report-out", help="Where to write the drift-report."
    ),
) -> None:
    """Monitor per-feature drift across cohorts and recommend a retrain (never auto-retrains)."""
    from mlfactory.domains.saas.monitor import monitor_drift

    cfg, df = _load(config)
    report = monitor_drift(df, cfg, threshold=threshold)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(report_out)

    if report.skipped:
        typer.echo("Drift monitoring unavailable — no usable date_col (snapshot mode).")
        return

    _sym = {"stable": "✔", "moderate": "⚠", "major": "✗"}
    typer.echo(
        f"Drift monitor  (reference {report.reference} → latest {report.latest}, "
        f"threshold {report.threshold:g})  → {report_out}"
    )
    typer.echo("")
    for r in report.features:
        typer.echo(f"  {_sym[r['status']]} {r['feature']:<24} PSI {r['psi']:.4f}  ({r['status']})")
    typer.echo("")
    if report.retrain_recommended:
        typer.echo(
            f"  ⚠ RETRAIN RECOMMENDED — {len(report.drifted)} feature(s) drifted past "
            f"{report.threshold:g}: {', '.join(report.drifted)}"
        )
        typer.echo("  (mlfactory proposes; the DS decides — it never auto-retrains)")
    else:
        typer.echo("  ✔ no significant drift — no retrain needed")


@app.command("train-uplift")
def train_uplift_cmd(
    data: Path = typer.Option(
        ..., "--data", help="A/B panel parquet (from `generate --treatment`)."
    ),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    learner: str = typer.Option("t", "--learner", help="Meta-learner: s | t."),
    model: str = typer.Option("logistic", "--model", help="Base model for the learner."),
    model_out: Path = typer.Option(
        Path("data/uplift.pkl"), "--model-out", help="Where to persist the fitted uplift model."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed (deterministic)."),
) -> None:
    """Fit an uplift meta-learner (target persuadables) on a randomized A/B panel."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.domains.saas.uplift import UpliftError, save_uplift, train_uplift

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    df = pd.read_parquet(data)
    try:
        um, card = train_uplift(df, cfg, learner=learner, base_model=model, seed=seed)
    except UpliftError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_uplift(um, model_out)
    card.write_json(model_out.with_suffix(".card.json"))

    name = {"s": "S-learner", "t": "T-learner"}[learner]
    typer.echo(f"{name} ({card.base_model}) on {card.n_train:,} rows → {model_out}")
    typer.echo(
        f"  treated {card.n_treated:,} / control {card.n_control:,} | "
        f"mean predicted uplift {card.ate_hat:+.4f}"
    )
    if card.tau_recovery_corr is not None:
        typer.echo(
            f"  recovery vs true uplift: corr {card.tau_recovery_corr:+.3f} "
            "(synthetic ground truth — how well it learned τ)"
        )


@app.command("uplift-eval")
def uplift_eval_cmd(
    model: Path = typer.Option(..., "--model", help="Fitted uplift model (from `train-uplift`)."),
    data: Path = typer.Option(..., "--data", help="A/B panel parquet to evaluate on."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    report_out: Path = typer.Option(
        Path("data/qini-report.json"), "--report-out", help="Where to write the qini-report."
    ),
) -> None:
    """Evaluate an uplift model: Qini coefficient, Qini curve, and uplift-by-decile."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.domains.saas.qini import QiniError, evaluate_uplift
    from mlfactory.domains.saas.uplift import load_uplift

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    um = load_uplift(model)
    df = pd.read_parquet(data)
    try:
        report = evaluate_uplift(um, df, cfg)
    except QiniError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(report_out)

    typer.echo(
        f"Uplift evaluation  ({report.n_rows:,} rows: {report.n_treated:,} treated / "
        f"{report.n_control:,} control)  → {report_out}"
    )
    typer.echo(f"\n  Qini coefficient: {report.qini_coefficient:.3f}  (>0 beats random targeting)")
    if report.tau_recovery_corr is not None:
        typer.echo(f"  τ recovery vs truth: corr {report.tau_recovery_corr:+.3f}")
    typer.echo("\n  observed uplift by predicted-uplift decile (1 = best-targeted):")
    for r in report.uplift_deciles:
        obs = f"{r['obs_uplift']:+.4f}" if r["obs_uplift"] is not None else "  n/a"
        typer.echo(f"    decile {r['decile']:>2}  n={r['n']:>6,}  observed uplift {obs}")


@app.command("policy-contrast")
def policy_contrast_cmd(
    model: Path = typer.Option(..., "--model", help="Risk model (.pkl, from `train`)."),
    uplift_model: Path = typer.Option(
        ..., "--uplift-model", help="Uplift model (.pkl, from `train-uplift`)."
    ),
    data: Path = typer.Option(..., "--data", help="A/B panel parquet (with `true_uplift`)."),
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    save_rate: float = typer.Option(0.3, "--save-rate", help="save_rate for the risk strategy."),
    offer_cost: float = typer.Option(5.0, "--offer-cost", help="Cost of one save-offer ($)."),
    budget: Optional[float] = typer.Option(None, "--budget", help="Total offer budget ($)."),
    n_offers: Optional[int] = typer.Option(None, "--n-offers", help="Max number of offers."),
    report_out: Path = typer.Option(
        Path("data/policy-contrast.json"), "--report-out", help="Where to write the contrast."
    ),
) -> None:
    """Head-to-head: target by risk vs by uplift at one budget, scored on the true effect."""
    import pandas as pd

    from mlfactory.config import ConfigError, load_config
    from mlfactory.compute.model import load_model
    from mlfactory.domains.saas.policy import PolicyError, contrast_policies
    from mlfactory.domains.saas.uplift import load_uplift

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    risk = load_model(model)
    up = load_uplift(uplift_model)
    df = pd.read_parquet(data)
    try:
        report = contrast_policies(
            risk,
            up,
            df,
            cfg,
            offer_cost=offer_cost,
            budget=budget,
            n_offers=n_offers,
            save_rate=save_rate,
        )
    except PolicyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(report_out)

    if budget is not None:
        limit = f"${budget:,.0f} budget"
    elif n_offers is not None:
        limit = f"{n_offers:,} offers"
    else:
        limit = "unlimited"
    typer.echo(
        f"Policy contrast — risk vs uplift  ({limit}, offer ${offer_cost:g}, "
        f"scored on the true counterfactual)  → {report_out}"
    )
    typer.echo("")
    typer.echo(
        f"  {'strategy':<9} {'targeted':>9} {'spend':>10} {'true net':>12} {'ROI':>7} {'sleeping dogs':>14}"
    )
    for name in ("risk", "uplift"):
        s = report.strategies[name]
        roi = f"{s['roi']:.2f}x" if s["roi"] is not None else "  n/a"
        typer.echo(
            f"  {name:<9} {s['n_targeted']:>9,} {'$' + format(s['spend'], ',.0f'):>10} "
            f"{'$' + format(s['true_net_value'], ',.0f'):>12} {roi:>7} {s['sleeping_dogs_treated']:>14,}"
        )
    typer.echo(
        f"\n  → targeting by uplift nets ${report.uplift_net_advantage:,.0f} more and treats "
        f"{report.sleeping_dogs_avoided:,} fewer sleeping dogs at the same budget."
    )


@app.command()
def advise(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (for a gate)."),
) -> None:
    """Print the copilot's pre-flight recommendations (features, split, policy) for your data."""
    from mlfactory.compute.profile import profile_frame
    from mlfactory.recommend import (
        recommend_experiment,
        recommend_features,
        recommend_policy,
        recommend_split,
    )

    cfg, df = _load(config)
    records = profile_frame(df, cfg)
    recs = [
        recommend_experiment(df),
        recommend_features(records),
        recommend_split(cfg),
        recommend_policy(cfg),
    ]

    if json_out:
        import json

        typer.echo(
            json.dumps({"command": "advise", "recommendations": [r.model_dump() for r in recs]})
        )
        return

    typer.echo(f"mlfactory advises  ({len(df):,} rows)\n")
    for r in recs:
        typer.echo(f"  [{r.gate}] {r.recommendation}")
        typer.echo(f"      why: {r.rationale}")
    typer.echo("\n  → `mlfactory run` acts on these with your approval at each step.")


@app.command()
def run(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    out_dir: Path = typer.Option(
        Path("data/run"), "--out-dir", help="Where to write splits, model, artifacts, report."
    ),
    models: str = typer.Option(
        "logistic,rf,xgboost", "--models", help="Model shortlist to compare."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Take every recommendation without prompting (non-interactive)."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
) -> None:
    """Interactive copilot — walk the pipeline, pausing at each gate to confirm or override."""
    from mlfactory.compute.compare import compare_models
    from mlfactory.compute.evaluate import evaluate_model
    from mlfactory.compute.model import MODELS, feature_columns, save_model, train_model
    from mlfactory.domains.saas.policy import simulate_policy
    from mlfactory.compute.profile import profile_frame
    from mlfactory.recommend import (
        recommend_features,
        recommend_model,
        recommend_policy,
        recommend_ship,
        recommend_split,
    )
    from mlfactory.report import build_html
    from mlfactory.compute.split import split_dataset

    def show(rec) -> None:
        typer.echo(f"\n▸ [{rec.gate}] {rec.recommendation}")
        typer.echo(f"    {rec.rationale}")

    cfg, df = _load(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "  (--yes: taking every recommendation)" if yes else ""
    typer.echo(f"mlfactory run — {len(df):,} rows{mode}")

    # ── Gate 1: features / leakage ──────────────────────────────────────
    frec = recommend_features(profile_frame(df, cfg))
    show(frec)
    excludes = frec.action["exclude"]
    if excludes and not (yes or typer.confirm(f"    exclude {', '.join(excludes)}?", default=True)):
        excludes = []
    numeric, categorical = feature_columns(df, cfg)
    cfg = cfg.model_copy(deep=True)
    cfg.columns.features = [c for c in numeric + categorical if c not in excludes]

    # ── Gate 2: split ───────────────────────────────────────────────────
    srec = recommend_split(cfg)
    show(srec)
    strategy = srec.action["strategy"]
    if not yes:
        strategy = typer.prompt("    split strategy", default=strategy)
    train_df, val_df, test_df, _ = split_dataset(df, cfg, strategy=strategy, seed=seed)
    typer.echo(f"    → train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    # ── Gate 3: model (stability over peak AUC) ─────────────────────────
    shortlist = [m.strip() for m in models.split(",") if m.strip()]
    rows = compare_models(train_df, val_df, cfg, models=shortlist, seed=seed)
    for r in rows:
        flag = " ✔stable" if r["stable"] else ""
        typer.echo(
            f"    {r['model']:<9} holdout AUC {r['holdout_auc']:.3f}  drop {r['auc_drop']:+.3f}{flag}"
        )
    mrec = recommend_model(rows)
    show(mrec)
    model = mrec.action["model"]
    if not yes:
        model = typer.prompt("    model to train", default=model)
    if model not in MODELS:
        typer.echo(f"unknown model {model!r} (use {' | '.join(MODELS)})")
        raise typer.Exit(code=1)
    est, card = train_model(train_df, cfg, model=model, seed=seed)
    save_model(est, out_dir / "model.pkl")
    card.write_json(out_dir / "model.card.json")

    # ── Gate 4: evaluate + ship read ────────────────────────────────────
    ev = evaluate_model(est, test_df, cfg, reference_df=train_df)
    ev.write_json(out_dir / "eval-report.json")
    typer.echo(f"    → held-out AUC {ev.metrics['auc']:.3f} | ECE {ev.metrics['ece']:.3f}")
    show(recommend_ship(ev.model_dump()))

    # ── Gate 5: policy ──────────────────────────────────────────────────
    show(recommend_policy(cfg))
    policy_report = None
    if cfg.columns.value_col is not None:
        save_rate, offer_cost, budget = 0.3, 5.0, None
        if not yes:
            save_rate = float(typer.prompt("    save_rate", default=0.3))
            offer_cost = float(typer.prompt("    offer_cost ($)", default=5.0))
            raw = typer.prompt("    budget ($, blank = unlimited)", default="")
            budget = float(raw) if str(raw).strip() else None
        policy_report = simulate_policy(
            est, test_df, cfg, save_rate=save_rate, offer_cost=offer_cost, budget=budget
        )
        policy_report.write_json(out_dir / "policy-report.json")
        roi = f", ROI {policy_report.roi:.2f}x" if policy_report.roi is not None else ""
        typer.echo(
            f"    → target {policy_report.n_targeted:,}, net ${policy_report.net_value:,.0f}{roi}"
        )

    # ── Report ──────────────────────────────────────────────────────────
    html = build_html(
        ev.model_dump(),
        policy_report.model_dump() if policy_report is not None else None,
        card.model_dump(),
    )
    (out_dir / "report.html").write_text(html)
    typer.echo(f"\n✔ done — model + artifacts + report in {out_dir}/  (open {out_dir}/report.html)")


@app.command("validate-artifact")
def validate_artifact_cmd(
    path: Path = typer.Argument(..., help="Artifact .md file to validate."),
    walk_lineage: bool = typer.Option(
        False, "--walk-lineage", help="Walk the parent lineage chain (sha + schema + status)."
    ),
    probe_output: bool = typer.Option(
        False, "--probe-output", help="Probe the declared on-disk output (rows + schema_hash)."
    ),
) -> None:
    """Validate an artifact: frontmatter schema + optional lineage walk + output probe (exit 1 on fail)."""
    from mlfactory.artifacts.validate import ValidationFailure, validate_artifact

    try:
        result = validate_artifact(path, walk_lineage=walk_lineage, probe_output=probe_output)
    except ValidationFailure as exc:
        typer.echo(f"✗ INVALID [{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any parse/schema error is a clean failure
        typer.echo(f"✗ INVALID {exc}", err=True)
        raise typer.Exit(code=1) from exc
    checks = (
        "schema" + (" + lineage" if walk_lineage else "") + (" + probe" if probe_output else "")
    )
    typer.echo(f"✔ {result['artifact']} valid ({checks}) — {path}")


@app.command("export-schemas")
def export_schemas_cmd(
    output_dir: Path = typer.Option(
        Path("schemas"), "--output-dir", help="Directory to write JSON-Schema into."
    ),
    check: bool = typer.Option(
        False, "--check", help="Verify on-disk schemas match the source; exit 1 on drift."
    ),
) -> None:
    """Emit JSON-Schema for every registered artifact model (or --check them for drift)."""
    from mlfactory.artifacts.schemas import ARTIFACT_MODELS, export_schemas

    drifted = export_schemas(output_dir, check=check)
    if check:
        if drifted:
            typer.echo(
                f"✗ schema drift: {', '.join(drifted)} — run export-schemas to regenerate", err=True
            )
            raise typer.Exit(code=1)
        typer.echo("✔ schemas in sync")
    else:
        typer.echo(f"Wrote {len(ARTIFACT_MODELS)} artifact schema(s) → {output_dir}")


@app.command("engineer-features")
def engineer_features_cmd(
    train: Path = typer.Option(
        ..., "--train", help="Train parquet (transforms are fit on this split)."
    ),
    spec: Path = typer.Option(
        ..., "--spec", help="Transforms YAML: {transforms: [ {id,name,type,inputs,...} ]}."
    ),
    output_dir: Path = typer.Option(
        Path("data/features"),
        "--output-dir",
        help="Where to write engineered splits + feature-spec.",
    ),
    val: Optional[Path] = typer.Option(None, "--val", help="Optional val parquet."),
    test: Optional[Path] = typer.Option(None, "--test", help="Optional test parquet."),
    json_out: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON summary."),
) -> None:
    """Engineer features (fit-on-train / apply-outward) and emit a feature-spec artifact."""
    import pandas as pd
    import yaml

    from mlfactory.artifacts.base import content_hash
    from mlfactory.artifacts.schemas import FeatureTransform
    from mlfactory.compute.engineer import (
        FeatureEngineeringError,
        build_feature_spec,
        engineer_features,
    )

    raw = yaml.safe_load(spec.read_text()) or {}
    try:
        transforms = [FeatureTransform.model_validate(t) for t in raw.get("transforms", [])]
    except Exception as exc:  # noqa: BLE001 — surface a bad spec as a clean error
        typer.echo(f"invalid feature spec: {exc}")
        raise typer.Exit(code=1) from exc

    tr = pd.read_parquet(train)
    va = pd.read_parquet(val) if val is not None else None
    te = pd.read_parquet(test) if test is not None else None
    try:
        frames, fit_params, produced = engineer_features(transforms, tr, val=va, test=te)
    except FeatureEngineeringError as exc:
        typer.echo(f"✗ {exc}")
        raise typer.Exit(code=1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = frames["train"]
    assert train_out is not None  # engineer_features always returns a train frame
    train_out.to_parquet(output_dir / "train.parquet", index=False)
    for split in ("val", "test"):
        frame = frames[split]
        if frame is not None:
            frame.to_parquet(output_dir / f"{split}.parquet", index=False)
    # declare the schema of what actually landed on disk (parquet round-trip is the source of truth)
    written_train = pd.read_parquet(output_dir / "train.parquet")
    artifact = build_feature_spec(
        transforms,
        written_train,
        fit_params,
        output_path="train.parquet",
        parent_sha256=content_hash(tr),
    )
    artifact.write_markdown(output_dir / "feature-spec.md")

    if json_out:
        import json

        typer.echo(
            json.dumps(
                {
                    "command": "engineer-features",
                    "output_dir": str(output_dir),
                    "n_transforms": len(transforms),
                    "train_rows": len(tr),
                    "produced": produced,
                    "feature_spec": str(output_dir / "feature-spec.md"),
                }
            )
        )
        return
    typer.echo(
        f"Engineered {len(transforms)} transform(s) on {len(tr):,} train rows → {output_dir}"
    )
    more = " …" if len(produced) > 8 else ""
    typer.echo(f"  produced {len(produced)} feature column(s): {', '.join(produced[:8])}{more}")
    typer.echo(
        f"  feature-spec.md written — validate with: "
        f"mlfactory validate-artifact {output_dir}/feature-spec.md --probe-output"
    )


@app.command("gen-model-card")
def gen_model_card_cmd(
    card: Path = typer.Option(..., "--card", help="model.card.json (from train)."),
    eval_report: Path = typer.Option(
        None, "--eval", help="Optional eval-report.json (from evaluate)."
    ),
    output: Path = typer.Option(
        Path("model-card.md"), "--output", help="Where to write the markdown card."
    ),
    target: str = typer.Option(
        "churn_next_30d", "--target", help="Target column (for the narrative)."
    ),
) -> None:
    """Render a markdown model card from the model + eval artifacts (the DS go/no-go surface)."""
    import json

    from mlfactory.model_card import gen_model_card

    mc = json.loads(card.read_text())
    ev = json.loads(eval_report.read_text()) if eval_report is not None else None
    md = gen_model_card(mc, ev, target=target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md)
    sections = [ln[3:] for ln in md.splitlines() if ln.startswith("## ")]
    typer.echo(f"Wrote model card → {output}  ({len(sections)} sections: {', '.join(sections)})")


@app.command("leakage-scan")
def leakage_scan_cmd(
    config: Path = typer.Option(
        Path("churn.yaml"), "--config", help="Path to the churn.yaml config."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Tier the target correlations into structured leakage risks (the deterministic EDA substrate)."""
    from mlfactory.compute.profile import profile_frame, scan_leakage

    cfg, df = _load(config)
    risks = scan_leakage(profile_frame(df, cfg), cfg)
    if json_out:
        import json

        typer.echo(json.dumps({"command": "leakage-scan", "leakage_risks": risks}))
        return
    if not risks:
        typer.echo("✔ no features cross the leakage tiers (|corr| ≥ 0.9).")
        return
    typer.echo(
        f"⚠ {len(risks)} leakage risk(s) — the EDA leakage-scanner should judge the posterior/derived cases:\n"
    )
    for r in risks:
        typer.echo(
            f"  [{r['kind']}] {r['column']}  strength {r['strength']:+.3f} → {r['recommendation']}"
        )
        typer.echo(f"      {r['reason']}")


if __name__ == "__main__":  # pragma: no cover
    app()
