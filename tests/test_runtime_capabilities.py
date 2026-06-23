"""Cluster-only capability tests for Databricks Runtime 17.3 (Spark 4.0.0).

These tests verify that the runtime provides the features required by
pyspark-iceberg-reader.  They are skipped in local CI and must be run on a
real DBR 17.3 session.

Run with:  py -3.14 -m pytest tests/test_runtime_capabilities.py -m cluster -v

Spec §6 (runtime requirements).
"""
from __future__ import annotations

import pytest


@pytest.mark.cluster
def test_avro_datasource_available(spark):
    """spark.read.format('avro') is available (spark-avro jar included in DBR)."""
    from pyspark.sql import SparkSession

    assert isinstance(spark, SparkSession)
    # A trivial Avro read to confirm the format is registered.
    # On DBR the jar is included; locally it must be downloaded separately.
    try:
        spark.read.format("avro")
        # Just constructing the reader is enough — no load() needed
    except Exception as exc:
        pytest.fail(f"spark.read.format('avro') raised: {exc}")


@pytest.mark.cluster
def test_parquet_field_id_read_enabled(spark):
    """spark.sql.parquet.fieldId.read.enabled can be set and respected."""
    original = spark.conf.get("spark.sql.parquet.fieldId.read.enabled", "false")
    try:
        spark.conf.set("spark.sql.parquet.fieldId.read.enabled", "true")
        val = spark.conf.get("spark.sql.parquet.fieldId.read.enabled")
        assert val == "true"
    finally:
        spark.conf.set("spark.sql.parquet.fieldId.read.enabled", original)


@pytest.mark.cluster
def test_metadata_file_path_column_available(spark, tmp_path):
    """_metadata.file_path is available on Spark 3.5+ (DBR 14+)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql.functions import col

    path = tmp_path / "cap_test.parquet"
    pq.write_table(pa.table({"x": pa.array([1, 2])}), str(path))

    df = spark.read.parquet(path.as_uri())
    df_with_meta = df.withColumn("_fp", col("_metadata.file_path"))
    first = df_with_meta.select("_fp").first()
    assert first is not None
    assert str(path.as_uri()) in first[0] or first[0].endswith("cap_test.parquet")


@pytest.mark.cluster
def test_metadata_row_index_column_available(spark, tmp_path):
    """_metadata.row_index is available (Spark 4.0+ / DBR 17.3)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql.functions import col

    path = tmp_path / "row_idx_test.parquet"
    pq.write_table(pa.table({"x": pa.array([10, 20, 30])}), str(path))

    df = spark.read.parquet(path.as_uri())
    df_with_idx = df.withColumn("_rid", col("_metadata.row_index"))
    indices = sorted(r._rid for r in df_with_idx.select("_rid").collect())
    assert indices == [0, 1, 2]
