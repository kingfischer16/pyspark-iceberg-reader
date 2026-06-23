"""Tests for position delete application (deletes.py).

Reads the committed copy_on_write fixture via spark.read.parquet (JVM-only,
no Python workers), extracts real _metadata.file_path / _metadata.row_index
values, then constructs a position-delete DataFrame via spark.sql() (also
JVM-only).  No spark.createDataFrame with data, no HADOOP_HOME needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

from pyspark_iceberg_reader.deletes import apply_position_deletes
from pyspark_iceberg_reader.planning.base import DeleteFile

_FIXTURES = Path(__file__).parent / "fixtures"
_COW_DIR = _FIXTURES / "copy_on_write" / "data"


@pytest.fixture(scope="module")
def cow_data(spark: SparkSession):
    """Read the copy-on-write fixture and add _fp, _rid, _seq helper columns."""
    path = (_COW_DIR / "part-0.parquet").as_uri()
    return (
        spark.read.parquet(path)
        .withColumn("_fp", col("_metadata.file_path"))
        .withColumn("_rid", col("_metadata.row_index"))
        .withColumn("_seq", lit(1))
    )


@pytest.fixture(scope="module")
def cow_file_path(cow_data):
    """Return the actual file path string baked into _metadata.file_path."""
    return cow_data.select("_fp").first()[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_position_deletes_returns_data_unchanged(cow_data, spark):
    result = apply_position_deletes(cow_data, [], spark)
    assert result.count() == cow_data.count()


def test_position_delete_removes_correct_row(cow_data, cow_file_path, spark, tmp_path):
    # Write a single-row position-delete Parquet that removes the first row (rid=0)
    import pyarrow as pa
    import pyarrow.parquet as pq

    del_path = tmp_path / "pos_delete.parquet"
    pq.write_table(
        pa.table({"file_path": [cow_file_path], "pos": pa.array([0], type=pa.int64())}),
        str(del_path),
    )

    delete_file = DeleteFile(
        path=del_path.as_uri(),
        file_format="PARQUET",
        content=1,
        spec_id=0,
        partition=(),
        data_sequence_number=1,
        equality_field_ids=(),
    )

    result = apply_position_deletes(cow_data, [delete_file], spark)

    original_ids = {r.id for r in cow_data.select("id").collect()}
    result_ids = {r.id for r in result.select("id").collect()}

    assert len(result_ids) == len(original_ids) - 1
    # The removed row is the one at row_index 0
    deleted_id = next(
        r.id for r in cow_data.orderBy("_rid").select("id", "_rid").collect() if r._rid == 0
    )
    assert deleted_id not in result_ids


def test_position_delete_wrong_path_does_not_remove_row(cow_data, spark, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    del_path = tmp_path / "pos_delete_wrong.parquet"
    pq.write_table(
        pa.table(
            {
                "file_path": ["s3://other-bucket/data/different.parquet"],
                "pos": pa.array([0], type=pa.int64()),
            }
        ),
        str(del_path),
    )

    delete_file = DeleteFile(
        path=del_path.as_uri(),
        file_format="PARQUET",
        content=1,
        spec_id=0,
        partition=(),
        data_sequence_number=1,
        equality_field_ids=(),
    )

    result = apply_position_deletes(cow_data, [delete_file], spark)
    assert result.count() == cow_data.count()


def test_position_delete_multiple_rows(cow_data, cow_file_path, spark, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Delete rows at positions 0 and 2
    del_path = tmp_path / "pos_delete_two.parquet"
    pq.write_table(
        pa.table(
            {
                "file_path": [cow_file_path, cow_file_path],
                "pos": pa.array([0, 2], type=pa.int64()),
            }
        ),
        str(del_path),
    )

    delete_file = DeleteFile(
        path=del_path.as_uri(),
        file_format="PARQUET",
        content=1,
        spec_id=0,
        partition=(),
        data_sequence_number=1,
        equality_field_ids=(),
    )

    result = apply_position_deletes(cow_data, [delete_file], spark)
    assert result.count() == cow_data.count() - 2


def test_position_delete_not_applied_when_delete_seq_less_than_data_seq(
    cow_data, cow_file_path, spark, tmp_path
):
    """Spec §7.6: delete only applies when delete.seq >= data.seq.
    A stale delete (seq=3) must not remove rows from a newer data file (seq=5).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    del_path = tmp_path / "stale_pos_delete.parquet"
    pq.write_table(
        pa.table({"file_path": [cow_file_path], "pos": pa.array([0], type=pa.int64())}),
        str(del_path),
    )

    # Override _seq to 5 to simulate a data file written after the delete file.
    data_newer = cow_data.withColumn("_seq", lit(5))

    delete_file = DeleteFile(
        path=del_path.as_uri(),
        file_format="PARQUET",
        content=1,
        spec_id=0,
        partition=(),
        data_sequence_number=3,  # delete seq=3 < data seq=5 → gate blocks it
        equality_field_ids=(),
    )

    result = apply_position_deletes(data_newer, [delete_file], spark)
    assert result.count() == cow_data.count()  # nothing removed
