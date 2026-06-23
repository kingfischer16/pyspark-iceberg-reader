"""Tests for equality delete application (deletes.py).

Reads the committed equality_deletes fixture (data + delete files) via
spark.read.parquet (JVM-only).  Helper columns (_seq, _spec_id, _partition)
are added via withColumn/lit (also JVM-only).  No spark.createDataFrame with
data, no Python workers needed, no HADOOP_HOME required.

Schema of equality_deletes/data/part-0.parquet: id (int64), name, category
Schema of equality_deletes/deletes/part-0.parquet: id (int64)
The delete file removes rows where id IN (2, 4).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, struct

from pyspark_iceberg_reader.deletes import apply_equality_deletes
from pyspark_iceberg_reader.planning.base import DeleteFile

_FIXTURES = Path(__file__).parent / "fixtures"
_DATA_PATH = (_FIXTURES / "equality_deletes" / "data" / "part-0.parquet").as_uri()
_DEL_PATH = (_FIXTURES / "equality_deletes" / "deletes" / "part-0.parquet").as_uri()

# Field IDs for the equality_deletes fixture schema:  id=1, name=2, category=3
# (Matches the schema we'll embed in the ScanPlan for tests.)
_SCHEMA_BY_ID = {1: "id", 2: "name", 3: "category"}

# For unpartitioned table: empty partition spec
_PARTITION_SPECS: dict = {}


def _eq_delete_file(
    path: str = _DEL_PATH,
    seq: int = 10,
    equality_ids: tuple = (1,),
) -> DeleteFile:
    return DeleteFile(
        path=path,
        file_format="PARQUET",
        content=2,
        spec_id=0,
        partition=(),
        data_sequence_number=seq,
        equality_field_ids=equality_ids,
    )


@pytest.fixture(scope="module")
def eq_data(spark: SparkSession):
    """Read equality-delete fixture data and add required helper columns."""
    return (
        spark.read.parquet(_DATA_PATH)
        # data_sequence_number = 5 (files were written at seq 5)
        .withColumn("_seq", lit(5))
        .withColumn("_spec_id", lit(0))
        .withColumn("_partition", struct())  # unpartitioned
        # Also add _fp and _rid so position-delete helpers in execution work
        .withColumn("_fp", col("_metadata.file_path"))
        .withColumn("_rid", col("_metadata.row_index"))
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_equality_deletes_returns_data_unchanged(eq_data, spark):
    result = apply_equality_deletes(eq_data, [], spark, _SCHEMA_BY_ID, _PARTITION_SPECS)
    assert result.count() == eq_data.count()


def test_equality_delete_removes_matching_rows(eq_data, spark):
    # Delete file has id IN (2, 4) with seq=10 > data._seq=5
    delete_file = _eq_delete_file(seq=10)

    result = apply_equality_deletes(eq_data, [delete_file], spark, _SCHEMA_BY_ID, _PARTITION_SPECS)

    result_ids = {r.id for r in result.select("id").collect()}
    assert 2 not in result_ids
    assert 4 not in result_ids
    assert 1 in result_ids
    assert 3 in result_ids
    assert 5 in result_ids
    assert result.count() == 3


def test_equality_delete_strict_seq_gate(eq_data, spark):
    # delete.seq == data._seq → NOT removed (spec: delete.seq must be STRICTLY GREATER)
    delete_file = _eq_delete_file(seq=5)  # same as data._seq

    result = apply_equality_deletes(eq_data, [delete_file], spark, _SCHEMA_BY_ID, _PARTITION_SPECS)

    # No rows removed because seq gate is strict (< not <=)
    assert result.count() == eq_data.count()


def test_equality_delete_null_matches_null(spark, tmp_path):
    """Null value in equality column matches null data row (spec §4.7)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Write a data Parquet with one null-id row
    data_path = tmp_path / "data_with_null.parquet"
    pq.write_table(
        pa.table({"id": pa.array([1, None, 3], type=pa.int64())}),
        str(data_path),
    )

    # Write an equality delete that matches null (id IS NULL)
    del_path = tmp_path / "eq_del_null.parquet"
    pq.write_table(
        pa.table({"id": pa.array([None], type=pa.int64())}),
        str(del_path),
    )

    data = (
        spark.read.parquet(data_path.as_uri())
        .withColumn("_seq", lit(1))
        .withColumn("_spec_id", lit(0))
        .withColumn("_partition", struct())
    )

    delete_file = _eq_delete_file(path=del_path.as_uri(), seq=5)
    result = apply_equality_deletes(data, [delete_file], spark, {1: "id"}, {})

    result_ids = {r.id for r in result.select("id").collect()}
    assert None not in result_ids
    assert 1 in result_ids
    assert 3 in result_ids
    assert result.count() == 2


def test_equality_delete_partition_scoped(spark, tmp_path):
    """Equality delete from spec_id=1 does not remove data from spec_id=0."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_path = tmp_path / "data_spec0.parquet"
    pq.write_table(
        pa.table({"id": pa.array([1, 2], type=pa.int64())}),
        str(data_path),
    )

    del_path = tmp_path / "eq_del_spec1.parquet"
    pq.write_table(
        pa.table({"id": pa.array([1], type=pa.int64())}),
        str(del_path),
    )

    data = (
        spark.read.parquet(data_path.as_uri())
        .withColumn("_seq", lit(1))
        .withColumn("_spec_id", lit(0))  # data is spec_id=0
        .withColumn("_partition", struct())
    )

    # Delete file claims spec_id=1 — different from data's spec_id=0
    delete_file = DeleteFile(
        path=del_path.as_uri(),
        file_format="PARQUET",
        content=2,
        spec_id=1,  # different spec!
        partition=(),
        data_sequence_number=5,
        equality_field_ids=(1,),
    )

    result = apply_equality_deletes(data, [delete_file], spark, {1: "id"}, {})

    # id=1 should NOT be removed because spec_ids don't match
    assert result.count() == 2
