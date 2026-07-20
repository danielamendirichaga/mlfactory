"""Data source loader — synthetic / file / sqlite / postgres, behind one interface.

The synthetic generator is just one more "source", so mlfactory is built and tested on
fake data and switched to a real database by changing one line of ``churn.yaml``.
:func:`load_data` returns a plain DataFrame regardless of where the data came from.

SQL is kept trivial (``SELECT * FROM <table>``) — per the "SQL is agnostic of logic"
principle, all real compute happens in Python on the extracted frame.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from mlfactory.config import ChurnConfig, SourceConfig


class SourceError(RuntimeError):
    """Raised when the configured data source cannot be loaded."""


def load_data(config: ChurnConfig) -> pd.DataFrame:
    """Load the dataset described by ``config.source``, coercing numeric-looking text columns."""
    src = config.source
    if src.kind == "synthetic":
        from mlfactory.domains.saas.generate import make_panel

        df = make_panel()
    elif src.kind == "file":
        df = _load_file(src)
    elif src.kind == "sqlite":
        df = _load_sqlite(src)
    elif src.kind == "postgres":
        df = _load_postgres(src)
    else:
        raise SourceError(f"Unknown source kind: {src.kind!r}")  # pragma: no cover
    return _coerce_numeric_like(df, config)


def _coerce_numeric_like(
    df: pd.DataFrame, config: ChurnConfig, threshold: float = 0.95
) -> pd.DataFrame:
    """Coerce object columns that are *mostly numeric text* (e.g. Telco's ``TotalCharges``) to numbers.

    Real CSVs routinely load a numeric column as text (a stray space, a thousands separator). We
    coerce a column only when ≥ ``threshold`` of its non-null values parse as numbers — a genuinely
    categorical column (``"Yes"``/``"No"``) is left alone — and skip the id/date/target columns. The
    coerced column names are recorded in ``df.attrs['coerced_numeric']`` so ``validate`` reports it.
    """
    cols = config.columns
    reserved = {cols.id_col, cols.date_col, cols.target_col}
    coerced: list[str] = []
    for c in df.columns:
        col = df[c]
        # skip the id/date/target and anything already numeric/bool/datetime — only text is a candidate
        if (
            c in reserved
            or pd.api.types.is_numeric_dtype(col)
            or pd.api.types.is_bool_dtype(col)
            or pd.api.types.is_datetime64_any_dtype(col)
        ):
            continue
        parsed = pd.to_numeric(col, errors="coerce")
        nonnull = int(col.notna().sum())
        if nonnull and int(parsed.notna().sum()) / nonnull >= threshold:
            df[c] = parsed
            coerced.append(c)
    df.attrs["coerced_numeric"] = coerced
    return df


def _load_file(src: SourceConfig) -> pd.DataFrame:
    assert src.path is not None  # guaranteed by config validation
    p = Path(src.path)
    if not p.exists():
        raise SourceError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in (".csv", ".txt"):
        return pd.read_csv(p)
    raise SourceError(f"Unsupported file extension {ext!r} (use .parquet or .csv): {p}")


def _load_sqlite(src: SourceConfig) -> pd.DataFrame:
    assert src.dsn is not None and src.table is not None  # guaranteed by config validation
    p = Path(src.dsn)
    if not p.exists():
        raise SourceError(f"SQLite database not found: {p}")
    con = sqlite3.connect(p)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{src.table}"', con)
    except Exception as exc:  # noqa: BLE001 — surface any DB error as a clean SourceError
        raise SourceError(f"Could not read table {src.table!r} from {p}: {exc}") from exc
    finally:
        con.close()


def _load_postgres(src: SourceConfig) -> pd.DataFrame:
    assert src.dsn is not None and src.table is not None  # guaranteed by config validation
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise SourceError(
            "postgres source requires SQLAlchemy — `pip install sqlalchemy psycopg2-binary`"
        ) from exc
    engine = create_engine(src.dsn)
    try:
        return pd.read_sql(f'SELECT * FROM "{src.table}"', engine)
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"Could not read table {src.table!r}: {exc}") from exc
    finally:
        engine.dispose()
