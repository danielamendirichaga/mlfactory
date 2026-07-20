"""Typed artifacts with lineage — the contract between pipeline stages (the *heavy* tier).

Every persisted stage emits a **typed** artifact recording what it produced, the params used, and
its **lineage** — so months later you can answer "what data produced this, and can I re-verify it?".
Downstream code reads known frontmatter fields, not free-form text.

This is the **heavy** contract tier: a rich :class:`ArtifactBase` (stage / verification / a lineage
``parent`` pointer / caveats), **markdown-with-frontmatter** serialization (YAML machine contract +
human body), plus the lineage/probe machinery in :mod:`mlfactory.artifacts.validate` and the
JSON-Schema exports in :mod:`mlfactory.artifacts.schemas`.

Backward compatibility: every new field is optional with a static/None default, so the medium-tier
artifacts (``split-manifest``, ``model-card``, …) that predate this upgrade keep validating and
round-tripping unchanged, and stay deterministic (no ``created_at`` is auto-stamped).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

# Verification methods a stage may use to prove its output before the artifact "counts".
VerificationMethod = Literal[
    "deterministic_script", "nbconvert_execute", "manual", "inline_probe", "viz_spec_validated"
]


class Parent(BaseModel):
    """Lineage pointer to the upstream artifact FILE this one was built from."""

    model_config = ConfigDict(extra="forbid")

    artifact: str  # the upstream artifact's type (e.g. "saved-dataset")
    path: str  # absolute or workspace-relative path to the parent artifact file
    sha256: str  # sha256 of the parent file's bytes (drift detection)
    version: str = "1.0"


class Verification(BaseModel):
    """The working-result gate: downstream refuses to build on a ``failed`` parent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "partial", "failed"] = "passed"
    method: VerificationMethod = "deterministic_script"
    ran_at: Optional[str] = None
    execution_log: Optional[str] = None
    errors: list[str] = []


class ArtifactBase(BaseModel):
    """Common base for every mlfactory artifact (heavy tier)."""

    model_config = ConfigDict(extra="forbid")

    artifact: str  # subclasses override with a Literal default (e.g. "split-manifest")
    version: str = "1.0"
    stage: Optional[int] = None
    created_at: Optional[str] = None  # ISO ts; caller-set (never auto-stamped → determinism)
    created_by: Optional[str] = None
    audience_mode: Optional[str] = None
    parent: Optional[Parent] = None  # THE lineage link (file pointer); null at a chain root
    parent_sha256: Optional[str] = None  # medium-tier compat: content hash of the input frame
    input_mode: Optional[Literal["from_artifact", "from_file", "cold"]] = None
    verification: Optional[Verification] = None
    backtrack_signals: list[dict] = []  # upstream re-entry requests
    caveats: list[str] = []

    # -- serialization ----------------------------------------------------- #
    def write_json(self, path: str | Path) -> None:
        """Write the artifact as a pretty JSON sidecar (medium-tier compatible)."""
        Path(path).write_text(self.model_dump_json(indent=2))

    def to_markdown(self, body: str = "") -> str:
        """Render as a markdown artifact: a YAML frontmatter contract + a human-readable body."""
        fm = yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        body = body.strip()
        return f"---\n{fm}---\n\n{body}\n" if body else f"---\n{fm}---\n"

    def write_markdown(self, path: str | Path, body: str = "") -> None:
        """Write the artifact as a ``.md`` file (frontmatter contract + body)."""
        Path(path).write_text(self.to_markdown(body))

    @staticmethod
    def split_frontmatter(text: str) -> tuple[dict, str]:
        """Parse a markdown artifact into (frontmatter dict, body). Frontmatter = the YAML
        between the first two ``---`` fences (the parse rule the whole contract rests on)."""
        if not text.lstrip().startswith("---"):
            raise ValueError("artifact has no YAML frontmatter (missing leading '---' fence)")
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError("artifact frontmatter is not closed by a second '---' fence")
        fm = yaml.safe_load(parts[1]) or {}
        body = parts[2].lstrip("\n")
        return fm, body

    @classmethod
    def from_markdown(cls, text: str) -> "ArtifactBase":
        """Validate a markdown artifact's frontmatter against this model."""
        fm, _ = cls.split_frontmatter(text)
        return cls.model_validate(fm)


# --------------------------------------------------------------------------- #
# deterministic fingerprints
# --------------------------------------------------------------------------- #
def content_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's content (row-order sensitive)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def schema_hash(df: pd.DataFrame) -> str:
    """Stable fingerprint of a frame's schema — sha256 over ``name:dtype`` per column, in order.

    Column order and dtype matter (a reorder or a dtype change bumps the hash), so a downstream
    stage can detect schema drift against a declared ``schema_hash`` without diffing bytes.
    """
    fp = "\n".join(f"{name}:{dtype}" for name, dtype in zip(df.columns, df.dtypes.astype(str)))
    return hashlib.sha256(fp.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's raw bytes (used to detect parent-artifact drift in the lineage walk)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
