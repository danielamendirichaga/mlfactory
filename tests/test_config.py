"""Tests for the churn.yaml config schema, loader, and the `init` template (S1)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.config import CONFIG_TEMPLATE, ChurnConfig, ConfigError, load_config

runner = CliRunner()


def _write(tmp_path, text: str):
    p = tmp_path / "churn.yaml"
    p.write_text(text)
    return p


MINIMAL = """
source:
  kind: synthetic
schema:
  id_col: subscriber_id
  target_col: churn_next_30d
"""


# --------------------------------------------------------------------------- #
# load_config — happy paths
# --------------------------------------------------------------------------- #
def test_minimal_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert isinstance(cfg, ChurnConfig)
    assert cfg.source.kind == "synthetic"
    assert cfg.columns.id_col == "subscriber_id"
    assert cfg.columns.target_col == "churn_next_30d"


def test_defaults_applied(tmp_path):
    """Omitted optionals get sensible defaults: no date_col, features='auto', positive=1."""
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.columns.date_col is None  # → snapshot mode downstream
    assert cfg.columns.value_col is None
    assert cfg.columns.features == "auto"
    assert cfg.columns.positive_value == 1


def test_explicit_feature_list_and_string_positive(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
source:
  kind: file
  path: data/telco.parquet
schema:
  id_col: customerID
  target_col: Churn
  positive_value: "Yes"
  features: [tenure, MonthlyCharges]
""",
        )
    )
    assert cfg.source.path == "data/telco.parquet"
    assert cfg.columns.positive_value == "Yes"
    assert cfg.columns.features == ["tenure", "MonthlyCharges"]


# --------------------------------------------------------------------------- #
# load_config — error paths (all raise ConfigError with a readable message)
# --------------------------------------------------------------------------- #
def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_missing_required_field_raises(tmp_path):
    cfg = """
source:
  kind: synthetic
schema:
  id_col: subscriber_id
"""  # no target_col
    with pytest.raises(ConfigError, match="target_col"):
        load_config(_write(tmp_path, cfg))


def test_unknown_field_rejected(tmp_path):
    cfg = MINIMAL + "  surprise_col: nope\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, cfg))


def test_bad_source_kind_rejected(tmp_path):
    cfg = """
source:
  kind: mongodb
schema:
  id_col: id
  target_col: churn
"""
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, cfg))


def test_file_source_requires_path(tmp_path):
    cfg = """
source:
  kind: file
schema:
  id_col: id
  target_col: churn
"""
    with pytest.raises(ConfigError, match="source.path"):
        load_config(_write(tmp_path, cfg))


def test_postgres_source_requires_dsn_and_table(tmp_path):
    cfg = """
source:
  kind: postgres
schema:
  id_col: id
  target_col: churn
"""
    with pytest.raises(ConfigError, match="dsn"):
        load_config(_write(tmp_path, cfg))


def test_not_a_mapping_raises(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))


# --------------------------------------------------------------------------- #
# init command — writes a template that itself loads (integration)
# --------------------------------------------------------------------------- #
def test_template_roundtrips(tmp_path):
    """The shipped CONFIG_TEMPLATE must itself be a valid config."""
    cfg = load_config(_write(tmp_path, CONFIG_TEMPLATE))
    assert cfg.source.kind == "synthetic"
    assert cfg.columns.date_col == "observation_month"


def test_init_writes_loadable_template(tmp_path):
    out = tmp_path / "churn.yaml"
    result = runner.invoke(app, ["init", "--path", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    # The file the CLI wrote loads cleanly.
    assert load_config(out).source.kind == "synthetic"


def test_init_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "churn.yaml"
    out.write_text("existing")
    result = runner.invoke(app, ["init", "--path", str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert out.read_text() == "existing"  # untouched

    forced = runner.invoke(app, ["init", "--path", str(out), "--force"])
    assert forced.exit_code == 0
    assert out.read_text() != "existing"
