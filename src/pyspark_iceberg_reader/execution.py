"""Shared execution engine: read data files, apply deletes, project to schema.

Both planners produce a ``ScanPlan``; this module is the single place that
turns it into a Spark ``DataFrame``.  All Spark operations stay in the JVM
(no Python workers needed for the core read path).

Spec §7.5.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, struct

from pyspark_iceberg_reader.deletes import (
    apply_equality_deletes,
    apply_position_deletes,
    _partition_struct_lit,
)
from pyspark_iceberg_reader.schema import iceberg_schema_to_spark

if TYPE_CHECKING:
    from pyspark_iceberg_reader.metadata import PartitionSpec
    from pyspark_iceberg_reader.paths import PathNormalizer
    from pyspark_iceberg_reader.planning.base import ScanPlan

log = logging.getLogger(__name__)

_FIELD_ID_CONF = "spark.sql.parquet.fieldId.read.enabled"


def _empty_dataframe(schema: "StructType", spark: SparkSession) -> DataFrame:
    return spark.range(0).select(
        *[lit(None).cast(f.dataType).alias(f.name) for f in schema.fields]
    )


def execute_scan_plan(
    plan: ScanPlan,
    spark: SparkSession,
    normalizer: PathNormalizer | None = None,
) -> DataFrame:
    """Execute a ``ScanPlan``: read data, apply deletes, project to current schema.

    Internal flow:
    1. Build a Spark read schema from ``plan.current_schema`` (field-ID metadata
       included so Spark matches columns by ID, not name).
    2. Enable ``spark.sql.parquet.fieldId.read.enabled`` if not already set.
    3. If the plan has no data files → return an empty DataFrame.
    4. Group data files by ``(seq, spec_id, partition)``; add ``_seq`` literal to
       every group so the position-delete sequence gate (spec §7.6) can operate.
       Add ``_spec_id`` and ``_partition`` only when equality deletes are present.
    5. Apply position deletes (seq-gated anti-join).
    6. Apply equality deletes if present (seq-gated, partition-scoped anti-join).
    7. Drop helper columns and project to ``plan.current_schema``.

    Spec §7.5.
    """
    read_schema = iceberg_schema_to_spark(plan.current_schema)
    output_cols = [f.name for f in plan.current_schema.fields]

    if not plan.data_files:
        return _empty_dataframe(read_schema, spark)

    # Spark evaluates this conf lazily at scan time, not at plan-build time,
    # so a try/finally restore would run before the DataFrame is consumed —
    # making field-ID reads silently fall back to name matching.  Set it here;
    # DBR sets it cluster-wide by default, so this is a no-op on real clusters.
    if spark.conf.get(_FIELD_ID_CONF, "false") != "true":
        spark.conf.set(_FIELD_ID_CONF, "true")

    has_equality = any(d.content == 2 for d in plan.delete_files)

    # Always group by (seq, spec_id, partition) so every data row carries _seq.
    # _seq is required by apply_position_deletes for the sequence gate (spec §7.6).
    groups: dict[tuple, list[str]] = {}
    for f in plan.data_files:
        key = (f.data_sequence_number, f.spec_id, f.partition)
        groups.setdefault(key, []).append(f.path)

    log.debug(
        "Reading %d data file(s) in %d group(s)%s.",
        len(plan.data_files),
        len(groups),
        " (equality deletes present)" if has_equality else "",
    )

    parts: list[DataFrame] = []
    for (seq, spec_id, partition), paths in groups.items():
        part = (
            spark.read.schema(read_schema).parquet(*paths)
            .withColumn("_fp", col("_metadata.file_path"))
            .withColumn("_rid", col("_metadata.row_index"))
            .withColumn("_seq", lit(seq))
        )
        if has_equality:
            spec = plan.partition_specs.get(spec_id)
            part = (
                part
                .withColumn("_spec_id", lit(spec_id))
                .withColumn("_partition", _partition_struct_lit(partition, spec))
            )
        parts.append(part)

    data = reduce(lambda a, b: a.unionByName(b), parts)

    data = apply_position_deletes(data, plan.delete_files, spark, normalizer)

    if has_equality:
        schema_by_id = {f.field_id: f.name for f in plan.current_schema.fields}
        data = apply_equality_deletes(
            data, plan.delete_files, spark, schema_by_id, plan.partition_specs, normalizer
        )

    return data.select(output_cols)
