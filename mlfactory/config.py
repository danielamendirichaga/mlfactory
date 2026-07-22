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

import json
import re
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


class ModelingDecisions(BaseModel):
    """Train & select knobs. Defaults reproduce today's hardcoded behavior (epic #17 / S0)."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: Literal["auc", "pr_auc", "ks", "top_decile_lift"] = "auc"
    imbalance: Literal["none", "smote"] = "none"
    calibrate: bool = False
    tune: Literal["none", "grid", "optuna"] = "none"
    penalty: Literal["l1", "l2", "elasticnet"] = (
        "l1"  # logistic regularization; l1 default (= prior behavior), now a per-dataset decision
    )
    max_auc_drop: float = 0.05  # compare._MAX_AUC_DROP (stability gate)
    max_score_psi: float = 0.2  # compare._MAX_SCORE_PSI (stability gate)


class EvaluationDecisions(BaseModel):
    """Held-out evaluation + ship knobs."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.5  # evaluate_model default operating point
    min_auc: float = 0.65  # recommend_ship discrimination floor
    max_ece: float = 0.10  # recommend_ship calibration bar
    segment_cols: list[str] | None = None  # None → auto (plan_tier, region)


class PolicyDecisions(BaseModel):
    """Downstream retention-policy economics."""

    model_config = ConfigDict(extra="forbid")

    save_rate: float = 0.3  # simulate-policy default
    offer_cost: float = 5.0  # simulate-policy default ($)
    budget: float | None = None
    targeting: Literal["risk", "uplift"] = "risk"


class MonitoringDecisions(BaseModel):
    """Drift-monitoring knobs."""

    model_config = ConfigDict(extra="forbid")

    drift_threshold: float = 0.25  # monitor_drift default PSI bar


class FeatureDecisions(BaseModel):
    """Feature-engineering approach — the FE gate's decision (epic #17 / S2b)."""

    model_config = ConfigDict(extra="forbid")

    approach: Literal["skip", "recipe", "hybrid"] = (
        "skip"  # skip = train on the raw split (default)
    )
    recipe_path: str | None = None  # the feature-spec YAML, when approach != "skip"


class CardDecisions(BaseModel):
    """DS-authored model-card sections — the card is authored, not just generated (epic #17 / S5)."""

    model_config = ConfigDict(extra="forbid")

    intended_use: str | None = None
    out_of_scope: str | None = None
    known_failure_modes: list[str] = []
    sign_off: str | None = None


class DecisionRecord(BaseModel):
    """The DS's confirmed choices for a run. Gates WRITE here; stages READ here.

    Every field defaults to the value the pipeline hardcoded before epic #17, so a config
    without a ``decisions:`` block behaves exactly as before. Each later slice (S3/S4/S6) swaps
    a hardcoded default for the matching field here — behavior-preserving until a gate overrides
    it. ``caveats`` accumulate across gates and propagate to the model card (S5).
    """

    model_config = ConfigDict(extra="forbid")

    modeling: ModelingDecisions = Field(default_factory=ModelingDecisions)
    evaluation: EvaluationDecisions = Field(default_factory=EvaluationDecisions)
    policy: PolicyDecisions = Field(default_factory=PolicyDecisions)
    monitoring: MonitoringDecisions = Field(default_factory=MonitoringDecisions)
    features: FeatureDecisions = Field(default_factory=FeatureDecisions)
    card: CardDecisions = Field(default_factory=CardDecisions)
    caveats: list[str] = []


class ChurnConfig(BaseModel):
    """The whole ``churn.yaml``: a data source, a column mapping, and the DS decision record."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: SourceConfig
    # YAML key is ``schema:``; exposed as ``.columns`` to avoid shadowing BaseModel.schema.
    columns: ColumnMap = Field(alias="schema")
    # DS decisions (metric/threshold/economics/…). Absent block → defaults = pre-#17 behavior.
    decisions: DecisionRecord = Field(default_factory=DecisionRecord)


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


def _set_exclude_columns_text(text: str, cols: list[str]) -> str:
    """Set ``schema.exclude_columns`` in a churn.yaml *text*, preserving comments.

    Replaces an existing (commented or active) ``exclude_columns:`` line under ``schema:``,
    or inserts one at the end of the schema block if none is present. A targeted line edit
    (not a re-serialize) so the template's onboarding comments survive.
    """
    flow = "[" + ", ".join(cols) + "]"
    lines = text.splitlines()
    schema_i = next((i for i, ln in enumerate(lines) if re.match(r"^schema\s*:", ln)), None)
    if schema_i is None:
        raise ConfigError("config has no top-level 'schema:' block to update")
    # the schema block runs until the next top-level key (column 0, non-comment) or EOF
    end = len(lines)
    for j in range(schema_i + 1, len(lines)):
        if re.match(r"^[^\s#]", lines[j]):
            end = j
            break
    exc_re = re.compile(r"^(\s*)#?\s*exclude_columns\s*:")
    for j in range(schema_i + 1, end):
        mm = exc_re.match(lines[j])
        if mm:
            lines[j] = f"{mm.group(1) or '  '}exclude_columns: {flow}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    # absent → insert at the end of the block, matching the block's child indent
    indent = "  "
    for j in range(schema_i + 1, end):
        m2 = re.match(r"^(\s+)\S", lines[j])
        if m2:
            indent = m2.group(1)
            break
    lines.insert(end, f"{indent}exclude_columns: {flow}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def set_exclude_columns(
    path: str | Path,
    *,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    replace: list[str] | None = None,
) -> list[str]:
    """Record a never-features (leakage) decision: update ``schema.exclude_columns`` in place.

    This is how a confirmed EDA leakage-drop actually reaches the pipeline. ``split``/``train``
    read ``config.columns.exclude_columns`` via :func:`~mlfactory.compute.model.feature_columns`,
    so a decision only takes effect once it is written *here* — the ``eda-exploration`` artifact
    recording it is not enough. The file is validated before and after; on any problem it is left
    untouched and :class:`ConfigError` is raised. Returns the new exclude list.
    """
    path = Path(path)
    cfg = load_config(path)  # validate + read the current list
    if replace is not None:
        new = list(dict.fromkeys(replace))
    else:
        new = list(dict.fromkeys(list(cfg.columns.exclude_columns) + list(add or [])))
        if remove:
            drop = set(remove)
            new = [c for c in new if c not in drop]
    updated = _set_exclude_columns_text(path.read_text(), new)
    try:  # never write a config we can't read back
        ChurnConfig.model_validate(yaml.safe_load(updated))
    except (ValidationError, yaml.YAMLError) as exc:
        raise ConfigError(f"Refusing to write an invalid config to {path}:\n{exc}") from exc
    path.write_text(updated)
    return new


def _write_decisions_block(text: str, decisions: dict) -> str:
    """Replace or append the tool-managed ``decisions:`` block in a churn.yaml *text*.

    The block carries no comments (it is machine-managed), so — unlike source/schema — it is safe
    to re-serialize wholesale; the source/schema blocks and their onboarding comments are untouched.
    """
    block = yaml.safe_dump({"decisions": decisions}, sort_keys=False, default_flow_style=False)
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^decisions\s*:", ln)), None)
    if start is None:
        return text.rstrip("\n") + "\n\n" + block  # append after a blank line
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^[^\s#]", lines[j]):
            end = j
            break
    new_lines = lines[:start] + block.rstrip("\n").splitlines() + lines[end:]
    return "\n".join(new_lines) + "\n"


def set_decision(path: str | Path, key: str, value: str) -> DecisionRecord:
    """Record a DS decision into the config's ``decisions:`` block.

    ``key`` is dotted (e.g. ``evaluation.threshold``, ``modeling.primary_metric``); ``value`` is
    coerced to the field's type by pydantic. An unknown key or an invalid value raises
    :class:`ConfigError` and the file is left untouched. This is the write half of the decision
    record — gates persist a confirmed choice here, the deterministic stages read it back from
    :attr:`ChurnConfig.decisions`. Returns the new record.
    """
    path = Path(path)
    cfg = load_config(path)
    data = cfg.decisions.model_dump()
    parts = key.split(".")
    node = data
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise ConfigError(f"unknown decision key: {key!r}")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"unknown decision key: {key!r}")
    # JSON-parse so lists / numbers / bools work (`[a, b]`, `0.3`, `true`); bare words
    # (`pr_auc`, `skip`) aren't valid JSON and fall through as plain strings.
    try:
        node[parts[-1]] = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        node[parts[-1]] = value
    try:
        record = DecisionRecord.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid decision {key}={value!r}:\n{exc}") from exc
    updated = _write_decisions_block(path.read_text(), record.model_dump())
    try:  # never write a config we can't read back
        ChurnConfig.model_validate(yaml.safe_load(updated))
    except (ValidationError, yaml.YAMLError) as exc:
        raise ConfigError(f"Refusing to write an invalid config to {path}:\n{exc}") from exc
    path.write_text(updated)
    return record


def write_source_schema(
    path: str | Path,
    *,
    source: SourceConfig,
    columns: ColumnMap,
) -> ChurnConfig:
    """Write the ``source`` + ``schema`` blocks to a churn.yaml (validated), preserving an existing
    ``decisions:`` block. The tested writer behind ``mlfactory configure`` / ``/mlfactory-setup`` — so the
    data mapping is set through validation, not free-hand YAML. Returns the written config; raises
    :class:`ConfigError` if the result can't be read back.
    """
    path = Path(path)
    decisions = DecisionRecord()
    if path.exists():
        try:
            decisions = load_config(path).decisions
        except ConfigError:
            decisions = DecisionRecord()
    cfg = ChurnConfig(
        source=source, columns=columns, decisions=decisions
    )  # validates the combination
    body = yaml.safe_dump(
        {
            "source": source.model_dump(exclude_none=True),
            "schema": columns.model_dump(exclude_none=True),
        },
        sort_keys=False,
        default_flow_style=False,
    )
    text = (
        "# mlfactory config — written by `mlfactory configure`.\n"
        "# Data mapping: edit via `configure`. Decisions: `record-decision` / `exclude-columns`.\n\n"
        + body
    )
    if decisions != DecisionRecord():
        text = _write_decisions_block(text, decisions.model_dump())
    try:  # never write a config we can't read back
        ChurnConfig.model_validate(yaml.safe_load(text))
    except (ValidationError, yaml.YAMLError) as exc:
        raise ConfigError(f"Refusing to write an invalid config to {path}:\n{exc}") from exc
    path.write_text(text)
    return cfg


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

# DS decisions (primary metric / operating threshold / policy economics / drift bar) live in a
# tool-managed `decisions:` block — set them with `mlfactory record-decision --key <k> --value <v>`
# (see them with `mlfactory decisions`). Omit it to use the defaults, which reproduce the
# pipeline's built-in behavior.
"""
