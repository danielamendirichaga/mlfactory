"""Config loading & validation — reads ``churn.yaml`` (data source + column mapping).

The config is what makes mlfactory domain-agnostic: the user *declares* their target /
date / id / feature columns here instead of the code hardcoding names. Everything downstream
reads a validated :class:`ChurnConfig`.

Public surface:

* :class:`ChurnConfig` (with :class:`SourceConfig` and :class:`ColumnMap`) — the typed schema.
* :func:`load_config` — read + validate a ``churn.yaml``; raises :class:`ConfigError` with a
  clear message on any problem.
* :data:`CONFIG_TEMPLATE` — the commented starter written by ``mlfactory init``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Raised when a ``churn.yaml`` is missing, unreadable, or invalid."""


class SourceConfig(BaseModel):
    """Where the data comes from. ``synthetic`` needs nothing else; the others do."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["postgres", "sqlite", "file", "synthetic"]
    dsn: str | None = None
    path: str | None = None
    table: str | None = None

    @model_validator(mode="after")
    def _require_fields_for_kind(self) -> SourceConfig:
        if self.kind in ("postgres", "sqlite"):
            missing = [f for f in ("dsn", "table") if not getattr(self, f)]
            if missing:
                raise ValueError(
                    f"source.{' and source.'.join(missing)} "
                    f"required when source.kind is '{self.kind}'"
                )
        elif self.kind == "file" and not self.path:
            raise ValueError("source.path is required when source.kind is 'file'")
        return self


class ColumnMap(BaseModel):
    """How mlfactory finds the columns it needs in *your* data.

    ``date_col`` is optional: present → panel path (time-aware split + drift monitoring);
    absent → single-snapshot path (drift/time-split skip gracefully). ``features='auto'``
    means "use every column except the id/date/target/value columns". ``exclude_columns``
    lists columns that must NEVER be used as features (leakage / experiment ground-truth /
    counterfactual oracle columns); the *domain* declares them, so the generic core hardcodes
    no domain column names.
    """

    model_config = ConfigDict(extra="forbid")

    id_col: str
    target_col: str
    positive_value: int | str | bool = 1
    date_col: str | None = None
    value_col: str | None = None
    features: list[str] | Literal["auto"] = "auto"
    exclude_columns: list[str] = []


class ChurnConfig(BaseModel):
    """The whole ``churn.yaml``: a data source + a column mapping."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: SourceConfig
    # YAML key is ``schema:``; exposed as ``.columns`` to avoid shadowing BaseModel.schema.
    columns: ColumnMap = Field(alias="schema")


def load_config(path: str | Path) -> ChurnConfig:
    """Read and validate a ``churn.yaml``.

    Raises :class:`ConfigError` (with a readable message) if the file is missing, is not a
    YAML mapping, or fails validation — so the CLI can print one clean line instead of a
    traceback.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}:\n{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config must be a YAML mapping (source: / schema:), got {type(raw).__name__} in {path}"
        )
    try:
        return ChurnConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid churn config ({path}):\n{exc}") from exc


CONFIG_TEMPLATE = """\
# mlfactory config — declares your data source and column mapping.
# Nothing is hardcoded; edit this to point mlfactory at your data.

source:
  kind: synthetic              # postgres | sqlite | file | synthetic
  # For file:      path: data/customers.parquet
  # For postgres:  dsn: "postgresql://user:pass@host/db"
  #                table: customers

schema:
  id_col: account_id        # unique customer / account id
  target_col: churn_next_30d   # the column to predict
  positive_value: 1            # value in target_col that means "churned"
  date_col: observation_month  # optional; enables time-aware split + drift. Omit for snapshot data.
  value_col: cltv              # optional; customer value for the policy simulator
  features: auto               # "auto" = all other columns, or a list: [tenure_months, product_usage_hours_30d]
  # exclude_columns: []        # never-features (leakage / A/B oracle cols, e.g. on a
  #   `generate --treatment` panel: [treated, true_uplift, churn_if_control, churn_if_treated])
"""
