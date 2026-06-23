"""End-to-end tests for execution.py: read, schema evolution, and delete application.

Uses committed Parquet fixtures (no Avro, no cluster) together with manually
constructed ScanPlan objects.  All Spark operations are JVM-only.

Fixture schemas (columns / pyarrow types):
  copy_on_write/data/part-0.parquet  : id int64, name string  (5 rows)
  equality_deletes/data/part-0.parquet : id int64, name string, category string (5 rows)
  equality_deletes/deletes/part-0.parquet : id int64  (rows with id=2 and id=4)
  schema_evolved/v1_data/part-0.parquet : id int32, name string  (rows id=1,2)
  schema_evolved/v2_data/part-0.parquet : id int64, name string, category string (rows id=3,4)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from pyspark_iceberg_reader.execution import execute_scan_plan
from pyspark_iceberg_reader.metadata import IcebergField, IcebergSchema
from pyspark_iceberg_reader.planning.base import DataFileTask, DeleteFile, ScanPlan

_FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_SCHEMA_ID_NAME = IcebergSchema(
    schema_id=1,
    fields=(
        IcebergField(field_id=1, name="id", field_type="long", required=False),
        IcebergField(field_id=2, name="name", field_type="string", required=False),
    ),
)

_SCHEMA_ID_NAME_CAT = IcebergSchema(
    schema_id=2,
    fields=(
        IcebergField(field_id=1, name="id", field_type="long", required=False),
        IcebergField(field_id=2, name="name", field_type="string", required=False),
        IcebergField(field_id=3, name="category", field_type="string", required=False),
    ),
)


def _data_task(path: Path, seq: int = 1, spec_id: int = 0) -> DataFileTask:
    return DataFileTask(
        path=path.as_uri(),
        file_format="PARQUET",
        spec_id=spec_id,
        partition=(),
        data_sequence_number=seq,
        record_count=0,
    )


def _pos_delete(path: Path, seq: int = 1) -> DeleteFile:
    return DeleteFile(
        path=path.as_uri(),
        file_format="PARQUET",
        content=1,
        spec_id=0,
        partition=(),
        data_sequence_number=seq,
        equality_field_ids=(),
    )


def _eq_delete(path: Path, seq: int = 10, eq_ids: tuple = (1,)) -> DeleteFile:
    return DeleteFile(
        path=path.as_uri(),
        file_format="PARQUET",
        content=2,
        spec_id=0,
        partition=(),
        data_sequence_number=seq,
        equality_field_ids=eq_ids,
    )


def _plan(schema, data_files, delete_files=None) -> ScanPlan:
    return ScanPlan(
        snapshot_id=1,
        current_schema=schema,
        partition_specs={},
        data_files=data_files,
        delete_files=delete_files or [],
    )


# ---------------------------------------------------------------------------
# Baseline: no deletes
# ---------------------------------------------------------------------------


def test_empty_plan_returns_empty_dataframe(spark: SparkSession):
    plan = _plan(_SCHEMA_ID_NAME, data_files=[])
    result = execute_scan_plan(plan, spark)
    assert result.count() == 0
    assert set(result.columns) == {"id", "name"}


def test_copy_on_write_no_deletes(spark: SparkSession):
    data_path = _FIXTURES / "copy_on_write" / "data" / "part-0.parquet"
    plan = _plan(_SCHEMA_ID_NAME, [_data_task(data_path)])

    result = execute_scan_plan(plan, spark)

    assert result.count() == 5
    assert set(result.columns) == {"id", "name"}
    ids = {r.id for r in result.select("id").collect()}
    assert ids == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Position deletes (end-to-end)
# ---------------------------------------------------------------------------


def test_position_delete_e2e(spark: SparkSession, tmp_path):
    """Position delete removes the correct row via _metadata.file_path / row_index."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_path = _FIXTURES / "copy_on_write" / "data" / "part-0.parquet"

    # Find the real file_path and row_index=0 from Spark metadata
    from pyspark.sql.functions import col

    raw = spark.read.parquet(data_path.as_uri())
    first_row = raw.withColumn("_fp", col("_metadata.file_path")).select("_fp").first()
    file_path_str = first_row[0]

    del_path = tmp_path / "pos_del.parquet"
    pq.write_table(
        pa.table({"file_path": [file_path_str], "pos": pa.array([0], type=pa.int64())}),
        str(del_path),
    )

    plan = _plan(
        _SCHEMA_ID_NAME,
        data_files=[_data_task(data_path, seq=1)],
        delete_files=[_pos_delete(del_path, seq=1)],
    )
    result = execute_scan_plan(plan, spark)

    assert result.count() == 4
    assert set(result.columns) == {"id", "name"}


# ---------------------------------------------------------------------------
# Equality deletes (end-to-end)
# ---------------------------------------------------------------------------


def test_equality_delete_e2e(spark: SparkSession):
    """Equality delete file (id IN 2,4) removes those rows from data."""
    data_path = _FIXTURES / "equality_deletes" / "data" / "part-0.parquet"
    del_path = _FIXTURES / "equality_deletes" / "deletes" / "part-0.parquet"

    plan = _plan(
        _SCHEMA_ID_NAME_CAT,
        data_files=[_data_task(data_path, seq=5)],
        delete_files=[_eq_delete(del_path, seq=10, eq_ids=(1,))],
    )
    result = execute_scan_plan(plan, spark)

    ids = {r.id for r in result.select("id").collect()}
    assert ids == {1, 3, 5}
    assert result.count() == 3
    assert set(result.columns) == {"id", "name", "category"}


def test_equality_delete_strict_seq_gate_e2e(spark: SparkSession):
    """delete.seq == data.seq → no rows removed (strict > required)."""
    data_path = _FIXTURES / "equality_deletes" / "data" / "part-0.parquet"
    del_path = _FIXTURES / "equality_deletes" / "deletes" / "part-0.parquet"

    plan = _plan(
        _SCHEMA_ID_NAME_CAT,
        data_files=[_data_task(data_path, seq=10)],  # same as delete seq
        delete_files=[_eq_delete(del_path, seq=10, eq_ids=(1,))],
    )
    result = execute_scan_plan(plan, spark)

    # delete.seq (10) is NOT > data._seq (10) → strict gate blocks removal
    assert result.count() == 5


# ---------------------------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------------------------


def test_schema_evolution_added_column_reads_null(spark: SparkSession):
    """Rows from v1_data (no 'category' column) read null for category."""
    v1_path = _FIXTURES / "schema_evolved" / "v1_data" / "part-0.parquet"

    # Read with current schema that includes 'category' (field_id=3)
    plan = _plan(_SCHEMA_ID_NAME_CAT, [_data_task(v1_path)])
    result = execute_scan_plan(plan, spark)

    assert result.count() == 2
    assert set(result.columns) == {"id", "name", "category"}
    categories = [r.category for r in result.select("category").collect()]
    assert all(c is None for c in categories)


def test_schema_evolution_type_promotion(spark: SparkSession):
    """v1_data has id as int32; current schema has id as long — reads correctly as long."""
    from pyspark.sql.types import LongType

    v1_path = _FIXTURES / "schema_evolved" / "v1_data" / "part-0.parquet"
    plan = _plan(_SCHEMA_ID_NAME, [_data_task(v1_path)])
    result = execute_scan_plan(plan, spark)

    id_type = dict(result.dtypes)["id"]
    assert id_type == "bigint"  # Spark's name for LongType
    assert result.count() == 2


def test_schema_evolution_multi_file_union(spark: SparkSession):
    """v1_data (no category) and v2_data (with category) unioned under current schema."""
    v1_path = _FIXTURES / "schema_evolved" / "v1_data" / "part-0.parquet"
    v2_path = _FIXTURES / "schema_evolved" / "v2_data" / "part-0.parquet"

    plan = _plan(
        _SCHEMA_ID_NAME_CAT,
        data_files=[_data_task(v1_path, seq=1), _data_task(v2_path, seq=2)],
    )
    result = execute_scan_plan(plan, spark)

    assert result.count() == 4
    rows = {r.id: r.category for r in result.select("id", "category").collect()}
    assert rows[1] is None   # v1_data has no category
    assert rows[2] is None
    assert rows[3] == "X"    # v2_data
    assert rows[4] == "Y"
