"""Deterministic synthetic **B2B SaaS** account-churn panel generator.

Pure synthetic numpy/pandas — **no real data, no PII**. Given a seed, :func:`make_panel`
produces exactly the same frame every time. This is mlfactory's bundled *reference domain*:
a generic B2B software product whose accounts are billed monthly, so the whole factory can be
exercised end-to-end on a realistic-but-safe dataset.

One row = one active **account** (a customer company) in one month (``observation_month``).
Accounts are simulated over their lifetime: they sign up, their product engagement evolves
monthly, and each month a latent-logit hazard decides whether they churn (cancel) in the next
30 days (``churn_next_30d``). A churned account's last row carries the label 1 and they leave.

The frame is intentionally imperfect (the four teaching levers):

* **Drift** — ``product_usage_hours_30d`` mean declines across ``observation_month`` cohorts
  (engagement erosion).
* **Imbalance** — ~8-12% monthly churn (intercept tuned by bisection).
* **Missingness** — product usage blank for brand-new accounts; ``company_size_employees`` ~8%.
* **Leakage trap** — ``cancel_page_visits_30d`` spikes with churn (an admin visiting the
  cancel/downgrade flow is a near-perfect *posterior* predictor; planted for the leakage scan).

Usage::

    python -m mlfactory.generate  # or: mlfactory generate --out data/saas_panel.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "churn_next_30d"
DRIFT_FEATURE = "product_usage_hours_30d"
COHORT_START = "2023-01"
TARGET_CHURN_RATE = 0.10

# v2 A/B-test simulation (only present when make_panel(treatment=True)).
# A randomized retention offer / CSM outreach to half the accounts.
TREATMENT_COL = "treated"
# Ground-truth columns — synthetic only, NEVER features (they encode the counterfactual).
ORACLE_COLS = ("true_uplift", "churn_if_control", "churn_if_treated")

PLANS = np.array(["Starter", "Growth", "Enterprise"])
PLAN_PROBS = [0.50, 0.35, 0.15]
PLAN_PRICE = {"Starter": 49, "Growth": 199, "Enterprise": 799}  # account MRR ($/month)
REGIONS = np.array(["namer", "emea", "apac", "latam"])
REGION_PROBS = [0.30, 0.30, 0.20, 0.20]
CHANNELS = np.array(["self_serve", "sales_led", "partner", "marketplace"])
CHANNEL_PROBS = [0.35, 0.35, 0.15, 0.15]
PAYMENT_METHODS = np.array(["credit_card", "invoice", "ach"])
PAYMENT_PROBS = [0.70, 0.20, 0.10]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _keep_mask(churn_row: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per account (contiguous blocks of length ``counts``), keep rows up to and
    including the first churn event; drop everything after (they have left)."""
    block_start = np.cumsum(counts) - counts
    cum_incl = np.cumsum(churn_row)
    cum_before_block = np.where(block_start > 0, cum_incl[block_start - 1], 0)
    within_incl = cum_incl - np.repeat(cum_before_block, counts)
    within_before = within_incl - churn_row
    return within_before == 0


def _uplift_tau(
    rng: np.random.Generator,
    discount_expiring: np.ndarray,
    days_since_login: np.ndarray,
    usage_trend: np.ndarray,
    support_tickets: np.ndarray,
    payment_failures: np.ndarray,
    total: int,
) -> np.ndarray:
    """Heterogeneous treatment effect τ(x) — the retention offer's reduction in churn probability.

    Built from a *different* signal than the churn hazard, so accounts at the **same risk**
    split into distinct uplift quadrants:

    * **persuadable** (τ large +): an expiring discount the offer can extend + mild, reversible
      product-usage disengagement → the offer genuinely rescues them;
    * **sleeping dog** (τ negative): billing/support friction → CSM outreach backfires
      (the "we noticed you've been quiet" note reminds them to cancel);
    * **lost cause** (τ ≈ 0): already gone (very high ``days_since_login``) → the offer is futile;
    * **sure thing** (τ ≈ 0): nothing pushing them out, nothing to rescue.

    ``tanh`` bounds τ to (−0.15, 0.15) churn-probability points; both signs occur by design.
    """
    persuade = (
        0.9 * discount_expiring.astype(float)  # biggest lever: extend an about-to-expire discount
        + 0.015 * np.clip(days_since_login, 0, 20)  # mildly disengaged = winnable
        + 0.02 * np.clip(-usage_trend, 0.0, None)  # a reversible slide in product usage
    )
    backfire = (
        0.10 * support_tickets  # already frustrated
        + 0.35 * payment_failures  # billing friction → contact backfires
        + 0.02 * np.clip(days_since_login - 25, 0, 5)  # too far gone → futile / irritating
    )
    raw = persuade - backfire + rng.normal(0.0, 0.1, total)
    return 0.15 * np.tanh(raw)


def make_panel(
    n_accounts: int = 8000,
    n_months: int = 24,
    seed: int = 42,
    treatment: bool = False,
) -> pd.DataFrame:
    """Build the synthetic B2B SaaS churn panel. Same ``(n_accounts, n_months, seed,
    treatment)`` → identical frame.

    With ``treatment=True`` (v2), overlay a **randomized A/B test**: a ``treated`` coin
    (~50/50, independent of features → a clean experiment) standing in for a retention offer /
    CSM outreach, a heterogeneous uplift ``τ(x)`` (see :func:`_uplift_tau`), and monotone-coupled
    potential outcomes so the observed ``churn_next_30d`` is the *factual* outcome under the
    assigned arm. Adds oracle columns (``true_uplift`` and both counterfactual outcomes) for
    honest Qini — never features. Default ``False`` leaves the v1 frame unchanged."""
    rng = np.random.default_rng(seed)
    n = n_accounts

    # --- static per-account traits ------------------------------------------
    # Signup month index; negative = signed up before the observation window.
    signup_idx = rng.integers(-11, n_months, size=n)
    plan = rng.choice(PLANS, size=n, p=PLAN_PROBS)
    mrr = np.array([PLAN_PRICE[p] for p in plan], dtype=np.int64)
    region = rng.choice(REGIONS, size=n, p=REGION_PROBS)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_PROBS)
    payment = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_PROBS)
    seats = rng.integers(1, 6, size=n)  # licensed seats (small teams)
    size_latent = np.clip(rng.normal(38.0, 12.0, n), 18, 80)  # firmographic latent
    engagement = np.clip(rng.lognormal(2.6, 0.5, n), 0.5, 80.0)  # base product-usage hrs/mo
    discount_len = rng.choice([0, 0, 0, 3, 6], size=n)  # months of intro discount (0 = none)
    size_missing = rng.random(n) < 0.08

    # Company size (employees): a lognormal-ish firmographic derived from the latent draw.
    company_size = np.clip(np.round(np.exp(size_latent / 14.0) + 1.0), 2, 5000)

    # --- build the full account-month grid (before churn truncation) ---------
    start = np.maximum(signup_idx, 0)
    counts = n_months - start
    total = int(counts.sum())
    acct = np.repeat(np.arange(n), counts)
    block_start_global = np.cumsum(counts) - counts
    offsets = np.arange(total) - np.repeat(block_start_global, counts)
    month_idx = np.repeat(start, counts) + offsets
    tenure = month_idx - np.repeat(signup_idx, counts) + 1  # months since signup (>=1)

    # --- dynamic per-row engagement features (as-of that month) --------------
    usage_mean = engagement[acct] * (1.0 - 0.02 * month_idx)  # DRIFT: declines by cohort
    usage = np.clip(rng.normal(usage_mean, np.maximum(usage_mean * 0.30, 0.5)), 0.0, 200.0)
    days_since = np.clip(
        np.round(30.0 / (1.0 + usage / 2.5) + rng.normal(0, 2, total)), 0, 30
    ).astype(np.int64)
    active_days = np.clip(
        np.round(30.0 * (1.0 - np.exp(-usage / 10.0)) + rng.normal(0, 2, total)), 0, 30
    ).astype(np.int64)
    key_actions = rng.poisson(np.clip(usage / 2.0, 0, 50)).astype(np.int64)
    features_adopted = rng.binomial(key_actions, 0.6)
    avg_session = np.clip(
        usage * 60.0 / np.maximum(active_days, 1) + rng.normal(0, 10, total), 1.0, 400.0
    )
    support_tickets = rng.poisson(0.20, total).astype(np.int64)
    payment_failures = rng.poisson(0.03, total).astype(np.int64)

    # usage_trend = change vs. previous month (0 in the first observed month)
    prev = np.empty(total)
    prev[1:] = usage[:-1]
    prev[block_start_global] = np.nan
    trend = np.where(np.isnan(prev), 0.0, usage - prev)

    discount_left = np.clip(np.repeat(discount_len, counts) - (tenure - 1), 0, None)
    discount_expiring = (np.repeat(discount_len, counts) > 0) & (discount_left <= 1)

    # --- latent churn hazard (on true, pre-missing values) -------------------
    base_logit = (
        -0.045 * (usage - 15.0)
        + 0.05 * days_since
        - 0.035 * tenure
        - 0.06 * trend
        + 0.70 * discount_expiring.astype(float)
        + 0.30 * support_tickets
        + 0.90 * payment_failures
        - 0.25 * (np.repeat(plan == "Enterprise", counts)).astype(float)
    )
    churn_u = rng.random(total)

    # Solve the intercept (bisection) so the kept-row churn rate ~= target.
    def realized_rate(intercept: float) -> float:
        churn_row = churn_u < _sigmoid(intercept + base_logit)
        keep = _keep_mask(churn_row, counts)
        kept = int(keep.sum())
        return float((keep & churn_row).sum() / kept) if kept else 0.0

    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if realized_rate(mid) > TARGET_CHURN_RATE:
            hi = mid
        else:
            lo = mid
    intercept = 0.5 * (lo + hi)

    p0 = _sigmoid(intercept + base_logit)  # churn prob under control
    if treatment:
        tau = _uplift_tau(
            rng, discount_expiring, days_since, trend, support_tickets, payment_failures, total
        )
        p1 = np.clip(p0 - tau, 0.0, 1.0)  # under treatment (τ>0 lowers churn; τ<0 raises it)
        treated = rng.random(total) < 0.5  # randomized, feature-independent
        # Monotone coupling: one uniform per row → clean individual-level effects.
        y0 = churn_u < p0
        y1 = churn_u < p1
        churn_bool = np.where(treated, y1, y0)  # observe only the factual arm
        true_uplift = p0 - p1  # retention uplift = churn-prob reduction
    else:
        churn_bool = churn_u < p0
    churn_row = churn_bool.astype(np.int64)
    keep = _keep_mask(churn_bool, counts)

    # --- assemble kept rows --------------------------------------------------
    base = pd.Period(COHORT_START, freq="M")
    min_idx = int(signup_idx.min())
    ext = pd.period_range(base + min_idx, periods=n_months - min_idx, freq="M")
    observation_month = ext[(month_idx - min_idx)].to_timestamp()
    signup_month = ext[(np.repeat(signup_idx, counts) - min_idx)].to_timestamp()

    csize = np.where(size_missing[acct], np.nan, company_size[acct])

    # Missingness on product usage for brand-new accounts (tenure == 1).
    new_missing = (tenure == 1) & (rng.random(total) < 0.5)
    usage_col = np.where(new_missing, np.nan, usage)
    avg_session_col = np.where(new_missing, np.nan, avg_session)

    # Leakage trap: visits to the cancel/downgrade flow spike for churners (with noise).
    cancel_visits = np.where(
        churn_row == 1, rng.poisson(4.0, total) + 1, rng.poisson(0.03, total)
    ).astype(np.int64)

    expected_life = np.clip(np.round(6.0 + 0.7 * engagement), 3, 48)[acct]
    cltv = (mrr[acct] * expected_life).astype(np.int64)

    data = {
        "account_id": (acct + 1).astype(np.int64),
        "observation_month": observation_month,
        "signup_month": signup_month,
        "tenure_months": tenure.astype(np.int64),
        "plan_tier": plan[acct],
        "mrr": mrr[acct],
        "payment_method": payment[acct],
        "in_discount": discount_left > 0,
        "discount_months_left": discount_left.astype(np.int64),
        "seats_licensed": seats[acct].astype(np.int64),
        "product_usage_hours_30d": usage_col.round(2),
        "active_days_30d": active_days,
        "days_since_last_login": days_since,
        "usage_trend_30d": trend.round(2),
        "key_actions_30d": key_actions,
        "features_adopted_30d": features_adopted,
        "avg_session_minutes": avg_session_col.round(1),
        "support_tickets_30d": support_tickets,
        "payment_failures_30d": payment_failures,
        "company_size_employees": csize,
        "region": region[acct],
        "acquisition_channel": channel[acct],
        "cancel_page_visits_30d": cancel_visits,
        "cltv": cltv,
        TARGET: churn_row,
    }
    if treatment:
        data[TREATMENT_COL] = treated.astype(np.int64)
        data["true_uplift"] = true_uplift.round(4)
        data["churn_if_control"] = y0.astype(np.int64)
        data["churn_if_treated"] = y1.astype(np.int64)

    df = pd.DataFrame(data).loc[keep]
    return df.sort_values(["account_id", "observation_month"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> str:
    """Human-readable summary (for the CLI, printed to stderr)."""
    first = df["observation_month"].min()
    last = df["observation_month"].max()
    null_rates = {c: round(float(df[c].isna().mean()), 4) for c in df.columns if df[c].isna().any()}
    lines = [
        f"rows            : {len(df):,}",
        f"accounts        : {df['account_id'].nunique():,}",
        f"cohorts         : {df['observation_month'].nunique()} months "
        f"({first:%Y-%m} → {last:%Y-%m})",
        f"churn rate      : {df[TARGET].mean():.4f} ({TARGET})",
        f"null rates      : {null_rates if null_rates else '{}'}",
        f"drift feature   : {DRIFT_FEATURE} (mean declines across cohorts)",
    ]
    if TREATMENT_COL in df.columns:
        treat_rate = df[TREATMENT_COL].mean()
        ate = float(df["churn_if_control"].mean() - df["churn_if_treated"].mean())
        sleeping = float((df["true_uplift"] < -0.01).mean())
        lines += [
            f"treated share   : {treat_rate:.3f} (randomized A/B — retention offer)",
            f"avg true uplift : {ate:+.4f} churn-prob reduction (ATE)",
            f"sleeping dogs   : {sleeping:.1%} of rows have negative uplift",
        ]
    return "\n".join(lines)
