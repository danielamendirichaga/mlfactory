"""Data validation — is the configured dataset *usable* by mlfactory?

:func:`validate` checks a loaded DataFrame against the declared :class:`ChurnConfig` and
returns a :class:`ValidationReport` — never a traceback. Findings are graded:

* **fail (✗)** — mlfactory cannot proceed (no target, empty, single-class, ...).
* **warn (⚠)** — proceed, but you should know (duplicate ids, a missing declared feature).
* **pass (✔)** — fine, with a short fact (e.g. the positive rate, the cohort count).

It also reports the **mode**: *panel* (a usable ``date_col`` → time-aware split + drift) or
*snapshot* (no date → those features skip gracefully).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlfactory.config import ChurnConfig
from mlfactory.domains.saas.generate import TREATMENT_COL

Status = Literal["pass", "warn", "fail"]
_SYMBOL: dict[Status, str] = {"pass": "✔", "warn": "⚠", "fail": "✗"}


@dataclass
class Check:
    name: str
    status: Status
    message: str


@dataclass
class ValidationReport:
    checks: list[Check]
    mode: str  # "panel" | "snapshot"

    @property
    def ok(self) -> bool:
        """True when there are no hard failures (mlfactory can proceed)."""
        return not any(c.status == "fail" for c in self.checks)

    def render(self) -> str:
        lines = [f"Validation ({self.mode} mode):", ""]
        lines += [f"  {_SYMBOL[c.status]} {c.message}" for c in self.checks]
        lines += ["", "Result: USABLE" if self.ok else "Result: NOT USABLE — fix the ✗ items."]
        return "\n".join(lines)


def validate(df: pd.DataFrame, config: ChurnConfig) -> ValidationReport:
    cols = config.columns
    present = set(df.columns)
    checks: list[Check] = []

    # rows
    if len(df) == 0:
        checks.append(Check("rows", "fail", "dataset is empty (0 rows)"))
    else:
        checks.append(Check("rows", "pass", f"{len(df):,} rows"))

    # target
    tgt = cols.target_col
    if tgt not in present:
        checks.append(Check("target", "fail", f"target column '{tgt}' not found"))
    else:
        values = set(df[tgt].dropna().unique())
        n_classes = len(values)
        if n_classes < 2:
            checks.append(Check("target", "fail", f"target '{tgt}' has {n_classes} class — need 2"))
        elif cols.positive_value not in values:
            checks.append(
                Check("target", "fail", f"positive_value {cols.positive_value!r} not in '{tgt}'")
            )
        else:
            rate = float((df[tgt] == cols.positive_value).mean())
            extra = (
                "" if n_classes == 2 else f" — but {n_classes} distinct values (expected binary)"
            )
            checks.append(
                Check(
                    "target",
                    "pass" if n_classes == 2 else "warn",
                    f"target '{tgt}': positive rate {rate:.1%}{extra}",
                )
            )

    # date / mode
    date_present = False
    if cols.date_col is None:
        checks.append(
            Check("date", "pass", "no date_col → snapshot mode (drift & time-aware split skipped)")
        )
    elif cols.date_col not in present:
        checks.append(Check("date", "fail", f"date_col '{cols.date_col}' declared but not found"))
    else:
        date_present = True
        n_cohorts = int(df[cols.date_col].nunique())
        checks.append(
            Check("date", "pass", f"date_col '{cols.date_col}' → panel mode ({n_cohorts} cohorts)")
        )
    mode = "panel" if date_present else "snapshot"

    # id + uniqueness (panel key is (id, date); snapshot key is id)
    idc = cols.id_col
    if idc not in present:
        checks.append(Check("id", "fail", f"id column '{idc}' not found"))
    elif date_present:
        dups = int(df.duplicated([idc, cols.date_col]).sum())  # type: ignore[list-item]
        if dups:
            checks.append(Check("id", "warn", f"id '{idc}': {dups} duplicate (id, date) rows"))
        else:
            checks.append(
                Check(
                    "id",
                    "pass",
                    f"id '{idc}': {df[idc].nunique():,} accounts, (id, date) unique",
                )
            )
    else:
        dups = int(df.duplicated([idc]).sum())
        if dups:
            checks.append(
                Check(
                    "id", "warn", f"id '{idc}': {dups} duplicate ids (expected unique in snapshot)"
                )
            )
        else:
            checks.append(Check("id", "pass", f"id '{idc}': {df[idc].nunique():,} unique"))

    # value_col (optional)
    if cols.value_col is not None and cols.value_col not in present:
        checks.append(
            Check("value", "warn", f"value_col '{cols.value_col}' not found — policy sim limited")
        )

    # numeric-looking text columns auto-coerced on load (set by load_data)
    coerced = df.attrs.get("coerced_numeric") or []
    if coerced:
        checks.append(Check("dtypes", "pass", f"auto-coerced text→numeric: {', '.join(coerced)}"))

    # experiment → is uplift (v2) available, or v1 (risk) only?
    if TREATMENT_COL in present and int(df[TREATMENT_COL].nunique(dropna=True)) >= 2:
        rate = float((df[TREATMENT_COL] == 1).mean())
        checks.append(
            Check(
                "experiment",
                "pass",
                f"'{TREATMENT_COL}' present ({rate:.0%} treated) → uplift (v2) available",
            )
        )
    else:
        checks.append(
            Check(
                "experiment",
                "pass",
                f"no '{TREATMENT_COL}' column → v1 (risk) pipeline; uplift (v2) needs a randomized A/B test",
            )
        )

    # features
    if cols.features != "auto":
        missing = [f for f in cols.features if f not in present]
        if missing:
            checks.append(Check("features", "warn", f"declared features not found: {missing}"))
        else:
            checks.append(
                Check("features", "pass", f"features: {len(cols.features)} declared, all present")
            )
    else:
        reserved = {idc, tgt, cols.date_col, cols.value_col}
        n_feat = len([c for c in present if c not in reserved])
        checks.append(Check("features", "pass", f"features: auto ({n_feat} columns)"))

    # missingness
    null = df.isna().mean()
    high = {c: round(float(v), 2) for c, v in null.items() if v > 0.5}
    if high:
        checks.append(Check("nulls", "warn", f"high missingness (>50%): {high}"))
    else:
        some = {c: round(float(v), 3) for c, v in null.items() if v > 0}
        checks.append(Check("nulls", "pass", f"nulls: {some}" if some else "no missing values"))

    return ValidationReport(checks=checks, mode=mode)
