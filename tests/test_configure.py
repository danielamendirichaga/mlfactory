"""Config setup (#34) — the tested `configure` writer + CLI.

`mlfactory configure` writes source + schema to churn.yaml through validation (not free-hand YAML),
preserving an existing `decisions:` block. It's the deterministic half of the `/mlfactory-setup` gate.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mlfactory.cli import app
from mlfactory.config import (
    ColumnMap,
    SourceConfig,
    load_config,
    set_decision,
    write_source_schema,
)

runner = CliRunner()


def test_configure_writes_a_valid_config(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    result = runner.invoke(
        app,
        [
            "configure",
            "--config",
            str(p),
            "--source-kind",
            "file",
            "--path",
            "data/x.parquet",
            "--target",
            "did_churn",
            "--id-col",
            "customer_id",
            "--date-col",
            "snapshot_date",
            "--value-col",
            "ltv",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = load_config(p)
    assert cfg.source.kind == "file" and cfg.source.path == "data/x.parquet"
    assert cfg.columns.target_col == "did_churn" and cfg.columns.id_col == "customer_id"
    assert cfg.columns.date_col == "snapshot_date" and cfg.columns.value_col == "ltv"
    assert cfg.columns.positive_value == 1  # "1" coerced to int
    assert cfg.columns.features == "auto"


def test_configure_features_list(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    result = runner.invoke(
        app,
        [
            "configure",
            "--config",
            str(p),
            "--source-kind",
            "file",
            "--path",
            "d.parquet",
            "--target",
            "y",
            "--id-col",
            "id",
            "--features",
            "a, b ,c",
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_config(p).columns.features == ["a", "b", "c"]


def test_configure_file_without_path_fails(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    result = runner.invoke(
        app,
        [
            "configure",
            "--config",
            str(p),
            "--source-kind",
            "file",
            "--target",
            "y",
            "--id-col",
            "id",
        ],
    )
    assert result.exit_code == 1  # SourceConfig: file requires path
    assert not p.exists()  # nothing written on failure


def test_configure_bad_source_kind_fails(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    result = runner.invoke(
        app,
        [
            "configure",
            "--config",
            str(p),
            "--source-kind",
            "mongodb",
            "--target",
            "y",
            "--id-col",
            "id",
        ],
    )
    assert result.exit_code == 1


def test_configure_positive_value_stays_a_string(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    result = runner.invoke(
        app,
        [
            "configure",
            "--config",
            str(p),
            "--source-kind",
            "file",
            "--path",
            "d.parquet",
            "--target",
            "status",
            "--id-col",
            "id",
            "--positive-value",
            "churned",
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_config(p).columns.positive_value == "churned"  # bare word, not JSON


def test_write_source_schema_preserves_decisions(tmp_path: Path) -> None:
    p = tmp_path / "churn.yaml"
    write_source_schema(
        p,
        source=SourceConfig(kind="file", path="d.parquet"),
        columns=ColumnMap(id_col="id", target_col="y"),
    )
    set_decision(p, "evaluation.threshold", "0.2")
    # re-configure (e.g. the data path changed) — the recorded decision must survive
    write_source_schema(
        p,
        source=SourceConfig(kind="file", path="d2.parquet"),
        columns=ColumnMap(id_col="id", target_col="y"),
    )
    cfg = load_config(p)
    assert cfg.source.path == "d2.parquet"
    assert cfg.decisions.evaluation.threshold == 0.2
