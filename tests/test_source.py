"""Tests for the data source loader (S3): synthetic / file / sqlite behind one interface."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from mlfactory.config import ChurnConfig
from mlfactory.source import SourceError, load_data

SCHEMA = {"id_col": "account_id", "target_col": "churn_next_30d"}

SMALL = pd.DataFrame(
    {
        "account_id": [1, 2, 3],
        "churn_next_30d": [0, 1, 0],
        "product_usage_hours_30d": [5.0, 1.0, 9.0],
    }
)


def _cfg(source: dict) -> ChurnConfig:
    return ChurnConfig.model_validate({"source": source, "schema": SCHEMA})


def test_synthetic_source():
    df = load_data(_cfg({"kind": "synthetic"}))
    assert len(df) > 0
    assert "churn_next_30d" in df.columns


def test_file_parquet(tmp_path):
    p = tmp_path / "data.parquet"
    SMALL.to_parquet(p, index=False)
    out = load_data(_cfg({"kind": "file", "path": str(p)}))
    pd.testing.assert_frame_equal(out, SMALL)


def test_file_csv(tmp_path):
    p = tmp_path / "data.csv"
    SMALL.to_csv(p, index=False)
    out = load_data(_cfg({"kind": "file", "path": str(p)}))
    assert list(out.columns) == list(SMALL.columns)
    assert len(out) == 3


def test_file_missing_raises(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        load_data(_cfg({"kind": "file", "path": str(tmp_path / "nope.parquet")}))


def test_file_bad_extension_raises(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}")
    with pytest.raises(SourceError, match="Unsupported file extension"):
        load_data(_cfg({"kind": "file", "path": str(p)}))


def test_sqlite_source(tmp_path):
    dbp = tmp_path / "customers.db"
    con = sqlite3.connect(dbp)
    SMALL.to_sql("customers", con, index=False)
    con.close()
    out = load_data(_cfg({"kind": "sqlite", "dsn": str(dbp), "table": "customers"}))
    assert list(out.columns) == list(SMALL.columns)
    assert len(out) == 3


def test_sqlite_missing_db_raises(tmp_path):
    with pytest.raises(SourceError, match="database not found"):
        load_data(_cfg({"kind": "sqlite", "dsn": str(tmp_path / "nope.db"), "table": "t"}))


# --- numeric-looking text is auto-coerced on load (S21) ------------------- #
def test_load_coerces_numeric_looking_text(tmp_path):
    n = 20
    raw = pd.DataFrame(
        {
            "account_id": range(n),
            "churn_next_30d": [0, 1] * (n // 2),
            # a stray " " keeps read_csv from auto-typing this numeric column (the Telco quirk)
            "spend": [f"{i}.5" for i in range(n - 1)] + [" "],
            "plan": ["Basic", "Premium"] * (n // 2),  # genuinely categorical — leave alone
        }
    )
    p = tmp_path / "raw.csv"
    raw.to_csv(p, index=False)
    out = load_data(_cfg({"kind": "file", "path": str(p)}))
    assert pd.api.types.is_numeric_dtype(out["spend"])  # coerced to numbers
    assert not pd.api.types.is_numeric_dtype(out["plan"])  # categorical text untouched
    assert out.attrs["coerced_numeric"] == ["spend"]  # and reported


def test_coerce_helper_threshold_blanks_and_exclusions():
    from mlfactory.source import _coerce_numeric_like

    n = 20
    df = pd.DataFrame(
        {
            "account_id": [str(i) for i in range(n)],  # numeric-text, but it's the id → excluded
            "churn_next_30d": [0, 1] * (n // 2),
            "spend": [f"{i}.0" for i in range(n - 1)] + [" "],  # 19/20 = 95% numeric → coerce
            "plan": ["a", "b"] * (n // 2),  # 0% numeric → keep
        }
    )
    out = _coerce_numeric_like(df, _cfg({"kind": "synthetic"}))
    assert pd.api.types.is_numeric_dtype(out["spend"]) and int(out["spend"].isna().sum()) == 1
    assert not pd.api.types.is_numeric_dtype(out["account_id"])  # id column never coerced
    assert not pd.api.types.is_numeric_dtype(out["plan"])  # categorical never coerced
    assert out.attrs["coerced_numeric"] == ["spend"]
