"""``validate-artifact`` — the lineage walker + on-disk probe (working-result enforcement).

The linchpin of the heavy contract tier. Two composable checks over a markdown artifact:

* :func:`walk_lineage` — walk the ``parent`` chain leaf→root; per upstream node, in this
  **deliberate order**: cycle → file-existence → sha256-drift → schema-validate → parent-type
  agreement → verification-status gate. A ``failed`` upstream verification poisons the chain.
* :func:`probe_output` — for an artifact declaring an on-disk parquet output, confirm the file
  exists, its row count matches, and its recomputed ``schema_hash`` matches the declared one.

The engine only *reports* (raises :class:`ValidationFailure` with a structured code, or returns a
verdict). The delete-on-failure rollback is the orchestrator's response to a non-zero exit — kept
out of here on purpose (the deterministic tool reports; the orchestrator reacts).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlfactory.artifacts.base import ArtifactBase, file_sha256, schema_hash
from mlfactory.artifacts.schemas import ARTIFACT_MODELS


class ValidationFailure(Exception):
    """A structured validation failure: a stable ``code`` + message + arbitrary details."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"{code}: {message}")


def _resolve(base_dir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p)


def _load(path: str | Path) -> tuple[dict, ArtifactBase]:
    """Load + schema-validate one markdown artifact; return (frontmatter, validated model)."""
    fm, _ = ArtifactBase.split_frontmatter(Path(path).read_text())
    model = ARTIFACT_MODELS.get(fm.get("artifact", ""), ArtifactBase)
    return fm, model.model_validate(fm)


def walk_lineage(path: str | Path) -> None:
    """Walk the parent chain leaf→root, raising :class:`ValidationFailure` on the first bad link."""
    seen: set[str] = set()
    _, art = _load(path)
    base_dir = Path(path).resolve().parent
    while art.parent is not None:
        p = art.parent
        parent_path = _resolve(base_dir, p.path)
        key = str(parent_path.resolve()) if parent_path.exists() else str(parent_path)
        # 1. cycle check — before hashing (a cycle can't be honestly hashed)
        if key in seen:
            raise ValidationFailure(
                "cycle_detected", f"parent already visited: {p.path}", path=p.path
            )
        seen.add(key)
        # 2. file existence
        if not parent_path.exists():
            raise ValidationFailure(
                "parent_file_missing", f"parent not found: {p.path}", path=p.path
            )
        # 3. sha256 drift
        actual = file_sha256(parent_path)
        if actual != p.sha256:
            raise ValidationFailure(
                "sha256_drift",
                f"parent bytes changed: {p.path}",
                path=p.path,
                declared=p.sha256,
                actual=actual,
            )
        # 4. schema-validate the parent frontmatter against its model
        try:
            _, parent_art = _load(parent_path)
        except Exception as exc:  # noqa: BLE001 — surface any parse/validation error as a code
            raise ValidationFailure(
                "parent_schema_invalid", f"{p.path}: {exc}", path=p.path
            ) from exc
        # 5. parent.artifact type agreement
        if parent_art.artifact != p.artifact:
            raise ValidationFailure(
                "parent_type_mismatch",
                f"declared {p.artifact!r}, found {parent_art.artifact!r}",
                path=p.path,
            )
        # 6. verification-status gate — a failed upstream poisons the chain
        if parent_art.verification is not None and parent_art.verification.status == "failed":
            raise ValidationFailure(
                "upstream_verification_failed", f"parent verification failed: {p.path}", path=p.path
            )
        art = parent_art
        base_dir = parent_path.resolve().parent


def probe_output(path: str | Path) -> None:
    """Probe an artifact's declared on-disk parquet output (existence + row count + schema_hash)."""
    fm, _ = _load(path)
    output = fm.get("output")
    if not output:
        return  # nothing to probe
    base_dir = Path(path).resolve().parent
    out_path = _resolve(base_dir, output["path"])
    if not out_path.exists():
        raise ValidationFailure(
            "output_file_missing",
            f"declared output not found: {output['path']}",
            path=output["path"],
        )
    df = pd.read_parquet(out_path)
    declared_rows = output.get("row_count")
    if declared_rows is not None and len(df) != declared_rows:
        raise ValidationFailure(
            "row_count_mismatch",
            f"declared {declared_rows}, found {len(df)}",
            declared=declared_rows,
            actual=len(df),
        )
    declared_hash = output.get("schema_hash")
    if declared_hash is not None:
        actual_hash = schema_hash(df)
        if actual_hash != declared_hash:
            raise ValidationFailure(
                "schema_hash_mismatch",
                "output schema drifted from declared",
                declared=declared_hash,
                actual=actual_hash,
            )


_WALK = walk_lineage
_PROBE = probe_output


def validate_artifact(
    path: str | Path, *, walk_lineage: bool = False, probe_output: bool = False
) -> dict:
    """Validate one artifact; raise :class:`ValidationFailure` on any problem, else return a verdict.

    Always schema-validates the leaf's own frontmatter. ``walk_lineage`` additionally walks the
    parent chain; ``probe_output`` additionally probes declared on-disk outputs.
    """
    _, art = _load(path)  # schema-validate the leaf
    if walk_lineage:
        _WALK(path)
    if probe_output:
        _PROBE(path)
    return {"valid": True, "artifact": art.artifact, "path": str(path)}
