"""Typed artifacts with lineage — the contract between pipeline steps.

Every persisted step (split, train, evaluate, policy, monitor) emits a small **typed** JSON
artifact recording what it produced, the params used, and its **lineage** (``parent_sha256``,
the content hash of the input it was built from). Downstream code reads known fields, not
free-form text; months later you can answer "what data produced this?".

This is the *medium* contract tier (see ADR-009): Pydantic models + lineage hashes + JSON
sidecars — no CI-synced JSON-Schema exports, versioning, or a lineage-walker (that's the
parked "heavy" tier).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict


class ArtifactBase(BaseModel):
    """Common base for every mlfactory artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact: str  # subclasses override with a Literal default (e.g. "split-manifest")
    parent_sha256: str | None = None  # lineage: content hash of the input this came from

    def write_json(self, path: str | Path) -> None:
        """Write the artifact as a pretty JSON sidecar."""
        Path(path).write_text(self.model_dump_json(indent=2))


def content_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's content (row-order sensitive)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()
