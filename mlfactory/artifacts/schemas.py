"""Registered heavy stage-artifact schemas + JSON-Schema export.

Holds the first heavy stage artifact — :class:`SavedDatasetArtifact`, the factory's **input
contract** (blueprint §7) — and the ``ARTIFACT_MODELS`` registry the lineage walker uses to
schema-validate each node by its declared type. :func:`export_schemas` emits (or ``--check``
verifies) a JSON-Schema per registered model, so the on-disk schemas can be CI-checked in sync
with the pydantic source.

Stage artifacts for the later pipeline stages (feature-spec, dataset, model) register here as
they are built — the walker and export stay generic over the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from mlfactory.artifacts.base import ArtifactBase


class SchemaColumn(BaseModel):
    """One column of a dataset's schema (``annotation`` is enriched later by EDA)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    nullable: bool = True
    annotation: Optional[str] = None


class DatasetOutput(BaseModel):
    """Where a dataset artifact's bytes live + the fingerprints a downstream stage re-checks."""

    model_config = ConfigDict(extra="forbid")

    path: str
    format: Literal["parquet", "csv", "feather"] = "parquet"
    row_count: int
    schema_hash: str
    size_bytes: Optional[int] = None


class SavedDatasetArtifact(ArtifactBase):
    """Stage-2 **input contract** — a versioned dataset handed to the factory (blueprint §7).

    Everything downstream depends only on this contract, not on how the bytes were produced.
    """

    artifact: Literal["saved-dataset"] = "saved-dataset"
    stage: int = 2
    output: DatasetOutput
    columns: list[SchemaColumn] = []


TransformType = Literal[
    "log_transform",
    "one_hot",
    "standard_scaler",
    "target_encoding",
    "date_parts",
    "temporal_diff",
    "drop_columns",
    "impute",
]


class FeatureTransform(BaseModel):
    """One entry in a feature-spec: a registry transform applied to inputs → output column(s)."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    type: TransformType
    inputs: list[str]
    params: dict = {}
    output_column: Optional[str] = None
    output_columns: Optional[list[str]] = None


class FeatureSpecArtifact(ArtifactBase):
    """Stage-4 output — the deterministic transform recipe + its fit-on-train learned params.

    ``fit_params`` records what each stateful transform learned on train (means, category sets,
    target-encoding maps …) so the exact transform can be replayed and audited.
    """

    artifact: Literal["feature-spec"] = "feature-spec"
    stage: int = 4
    transforms: list[FeatureTransform]
    fit_params: dict = {}
    output: DatasetOutput
    target_compatibility: list[dict] = []


# Registry: artifact-type string -> its pydantic model. The lineage walker validates each node
# against this; export_schemas emits one JSON-Schema per entry. Later stages append their models.
ARTIFACT_MODELS: dict[str, type[ArtifactBase]] = {
    "saved-dataset": SavedDatasetArtifact,
    "feature-spec": FeatureSpecArtifact,
}


def export_schemas(output_dir: str | Path, *, check: bool = False) -> list[str]:
    """Emit (or, with ``check=True``, verify) JSON-Schema for every registered artifact model.

    Returns the list of artifact types whose on-disk schema **drifted** from the pydantic source
    (empty ⇒ in sync). Writes ``<type>.schema.json`` per model when not checking.
    """
    out = Path(output_dir)
    drifted: list[str] = []
    for name, model in sorted(ARTIFACT_MODELS.items()):
        schema = json.dumps(model.model_json_schema(), indent=2, sort_keys=True)
        target = out / f"{name}.schema.json"
        if check:
            if not target.exists() or target.read_text() != schema:
                drifted.append(name)
        else:
            out.mkdir(parents=True, exist_ok=True)
            target.write_text(schema)
    return drifted
