"""Tests for validate-artifact — the lineage walker, the on-disk probe, and schema export."""

from __future__ import annotations

import pandas as pd
import pytest

from mlfactory.artifacts.base import Parent, Verification, file_sha256, schema_hash
from mlfactory.artifacts.schemas import DatasetOutput, SavedDatasetArtifact, export_schemas
from mlfactory.artifacts.validate import (
    ValidationFailure,
    probe_output,
    validate_artifact,
    walk_lineage,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _write_saved(dirpath, name, df, *, parent=None, status="passed"):
    """Write a saved-dataset .md + its parquet under dirpath; return the .md path."""
    df.to_parquet(dirpath / f"{name}.parquet", index=False)
    art = SavedDatasetArtifact(
        verification=Verification(status=status),
        parent=parent,
        output=DatasetOutput(
            path=f"{name}.parquet", row_count=len(df), schema_hash=schema_hash(df)
        ),
    )
    md = dirpath / f"{name}.saved-dataset.md"
    art.write_markdown(md)
    return md


def _parent_to(md_path):
    return Parent(artifact="saved-dataset", path=md_path.name, sha256=file_sha256(md_path))


# --- lineage walk: happy path + each failure code ------------------------- #
def test_valid_chain_passes(tmp_path, df):
    root = _write_saved(tmp_path, "root", df)
    child = _write_saved(tmp_path, "child", df, parent=_parent_to(root))
    result = validate_artifact(child, walk_lineage=True, probe_output=True)
    assert result["valid"] and result["artifact"] == "saved-dataset"


def test_parent_file_missing(tmp_path, df):
    parent = Parent(artifact="saved-dataset", path="nope.saved-dataset.md", sha256="0" * 64)
    leaf = _write_saved(tmp_path, "leaf", df, parent=parent)
    with pytest.raises(ValidationFailure) as e:
        walk_lineage(leaf)
    assert e.value.code == "parent_file_missing"


def test_sha256_drift(tmp_path, df):
    _write_saved(tmp_path, "root", df)  # parent must exist so the walk reaches the sha check
    parent = Parent(artifact="saved-dataset", path="root.saved-dataset.md", sha256="0" * 64)
    leaf = _write_saved(tmp_path, "leaf", df, parent=parent)
    with pytest.raises(ValidationFailure) as e:
        walk_lineage(leaf)
    assert e.value.code == "sha256_drift"


def test_parent_type_mismatch(tmp_path, df):
    root = _write_saved(tmp_path, "root", df)
    parent = Parent(artifact="feature-spec", path="root.saved-dataset.md", sha256=file_sha256(root))
    leaf = _write_saved(tmp_path, "leaf", df, parent=parent)
    with pytest.raises(ValidationFailure) as e:
        walk_lineage(leaf)
    assert e.value.code == "parent_type_mismatch"


def test_upstream_verification_failed(tmp_path, df):
    root = _write_saved(tmp_path, "root", df, status="failed")  # poisoned upstream
    leaf = _write_saved(tmp_path, "leaf", df, parent=_parent_to(root))
    with pytest.raises(ValidationFailure) as e:
        walk_lineage(leaf)
    assert e.value.code == "upstream_verification_failed"


def test_cycle_detected(tmp_path, df, monkeypatch):
    # A honestly-hashed cycle is impossible to construct (that IS the guarantee — the cycle check
    # runs before hashing). Neutralize the sha check to prove the cycle guard itself fires.
    import mlfactory.artifacts.validate as V

    a = _write_saved(
        tmp_path,
        "a",
        df,
        parent=Parent(artifact="saved-dataset", path="b.saved-dataset.md", sha256="x"),
    )
    _write_saved(
        tmp_path,
        "b",
        df,
        parent=Parent(artifact="saved-dataset", path="a.saved-dataset.md", sha256="x"),
    )
    monkeypatch.setattr(V, "file_sha256", lambda _p: "x")
    with pytest.raises(ValidationFailure) as e:
        V.walk_lineage(a)
    assert e.value.code == "cycle_detected"


# --- output probe --------------------------------------------------------- #
def test_probe_passes(tmp_path, df):
    probe_output(_write_saved(tmp_path, "leaf", df))  # no raise


def test_probe_row_count_mismatch(tmp_path, df):
    leaf = _write_saved(tmp_path, "leaf", df)
    df.iloc[:2].to_parquet(tmp_path / "leaf.parquet", index=False)  # fewer rows than declared
    with pytest.raises(ValidationFailure) as e:
        probe_output(leaf)
    assert e.value.code == "row_count_mismatch"


def test_probe_schema_hash_mismatch(tmp_path, df):
    leaf = _write_saved(tmp_path, "leaf", df)
    df.rename(columns={"a": "z"}).to_parquet(tmp_path / "leaf.parquet", index=False)  # schema drift
    with pytest.raises(ValidationFailure) as e:
        probe_output(leaf)
    assert e.value.code == "schema_hash_mismatch"


def test_probe_output_missing(tmp_path, df):
    leaf = _write_saved(tmp_path, "leaf", df)
    (tmp_path / "leaf.parquet").unlink()
    with pytest.raises(ValidationFailure) as e:
        probe_output(leaf)
    assert e.value.code == "output_file_missing"


# --- schema export -------------------------------------------------------- #
def test_export_schemas_roundtrip_and_drift(tmp_path):
    assert export_schemas(tmp_path) == []
    assert (tmp_path / "saved-dataset.schema.json").exists()
    assert export_schemas(tmp_path, check=True) == []  # in sync
    (tmp_path / "saved-dataset.schema.json").write_text("{}")  # tamper
    assert export_schemas(tmp_path, check=True) == ["saved-dataset"]


# --- CLI (behavioral) ----------------------------------------------------- #
def test_cli_validate_artifact(tmp_path, df):
    from typer.testing import CliRunner

    from mlfactory.cli import app

    leaf = _write_saved(tmp_path, "leaf", df)
    ok = CliRunner().invoke(app, ["validate-artifact", str(leaf), "--probe-output"])
    assert ok.exit_code == 0 and "valid" in ok.output

    (tmp_path / "leaf.parquet").unlink()  # break the output
    bad = CliRunner().invoke(app, ["validate-artifact", str(leaf), "--probe-output"])
    assert bad.exit_code == 1
