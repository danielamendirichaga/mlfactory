"""Tests for the heavy artifact base (frontmatter serialization, fingerprints, backward compat)."""

from __future__ import annotations

import hashlib

import pandas as pd

from mlfactory.artifacts import (
    ArtifactBase,
    Verification,
    content_hash,
    file_sha256,
    schema_hash,
)
from mlfactory.artifacts.schemas import DatasetOutput, SavedDatasetArtifact


def _saved(df: pd.DataFrame) -> SavedDatasetArtifact:
    return SavedDatasetArtifact(
        verification=Verification(status="passed"),
        output=DatasetOutput(path="d.parquet", row_count=len(df), schema_hash=schema_hash(df)),
    )


# --- markdown-with-frontmatter serialization ------------------------------ #
def test_frontmatter_roundtrip():
    art = _saved(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    md = art.to_markdown("# Human body\nSome narrative for a reviewer.")
    assert md.startswith("---")
    fm, body = ArtifactBase.split_frontmatter(md)
    assert fm["artifact"] == "saved-dataset" and fm["stage"] == 2
    assert "Human body" in body
    assert SavedDatasetArtifact.from_markdown(md) == art  # frontmatter round-trips exactly


def test_write_markdown(tmp_path):
    art = _saved(pd.DataFrame({"a": [1, 2]}))
    out = tmp_path / "a.saved-dataset.md"
    art.write_markdown(out, "body text")
    assert SavedDatasetArtifact.from_markdown(out.read_text()) == art


def test_no_frontmatter_raises():
    import pytest

    with pytest.raises(ValueError, match="frontmatter"):
        ArtifactBase.split_frontmatter("no fences here")


# --- deterministic fingerprints ------------------------------------------- #
def test_schema_hash_deterministic_and_sensitive():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    assert schema_hash(df) == schema_hash(df.copy())  # deterministic
    assert schema_hash(df) != schema_hash(df.rename(columns={"a": "z"}))  # name change
    assert schema_hash(df) != schema_hash(df.assign(a=df["a"].astype(float)))  # dtype change
    assert schema_hash(df) != schema_hash(df[["b", "a"]])  # column order matters


def test_content_hash_row_order_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert content_hash(df) == content_hash(df.copy())
    assert content_hash(df) != content_hash(df.iloc[::-1])


def test_file_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert file_sha256(p) == hashlib.sha256(b"hello").hexdigest()


# --- backward compatibility with the medium-tier artifacts ---------------- #
def test_default_heavy_artifact_is_deterministic():
    # No created_at is auto-stamped → two identical artifacts are equal (reproducibility preserved).
    out = DatasetOutput(path="x.parquet", row_count=1, schema_hash="h")
    a = SavedDatasetArtifact(output=out)
    b = SavedDatasetArtifact(output=out)
    assert a == b and a.created_at is None and a.version == "1.0"


def test_existing_medium_artifact_still_roundtrips_json(tmp_path):
    """A pre-existing medium-tier artifact still round-trips as JSON and gains heavy fields as defaults."""
    from mlfactory.compute.split import SplitManifest, split_dataset
    from mlfactory.config import ChurnConfig
    from mlfactory.domains.saas.generate import make_panel

    cfg = ChurnConfig.model_validate(
        {
            "source": {"kind": "synthetic"},
            "schema": {
                "id_col": "account_id",
                "target_col": "churn_next_30d",
                "date_col": "observation_month",
            },
        }
    )
    df = make_panel(n_accounts=300, n_months=8, seed=1)
    *_, manifest = split_dataset(df, cfg, strategy="time")
    p = tmp_path / "m.json"
    manifest.write_json(p)
    assert SplitManifest.model_validate_json(p.read_text()) == manifest
    # gained the heavy-tier fields as inert defaults
    assert manifest.version == "1.0" and manifest.parent is None and manifest.caveats == []
    # and can now render as a markdown artifact too
    assert "artifact: split-manifest" in manifest.to_markdown()
