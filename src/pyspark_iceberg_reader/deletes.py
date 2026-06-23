"""Position and equality delete application helpers.

Both functions accept a data ``DataFrame`` that already carries helper columns
added by ``execution.py``:
- Position deletes need ``_fp`` (file path) and ``_rid`` (row index).
- Equality deletes need ``_seq`` (data sequence number), ``_spec_id``,
  and ``_partition`` (struct column built from the partition spec).

Helper columns are dropped before returning — the caller projects to the
final schema.  Spec §4.7, §7.6, §7.7.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import broadcast, col, lit, struct, when, regexp_replace

from pyspark_iceberg_reader.errors import UnsupportedFeatureError
from pyspark_iceberg_reader.paths import PathNormalizer, normalize_path

if TYPE_CHECKING:
    from pyspark_iceberg_reader.metadata import PartitionSpec
    from pyspark_iceberg_reader.planning.base import DeleteFile

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position deletes  (spec §7.6)
# ---------------------------------------------------------------------------


def apply_position_deletes(
    data: DataFrame,
    delete_files: list[DeleteFile],
    spark: SparkSession,
    normalizer: PathNormalizer | None = None,
) -> DataFrame:
    """Remove rows identified by position delete files.

    *data* must carry ``_fp`` (``_metadata.file_path`` value), ``_rid``
    (``_metadata.row_index`` value), and ``_seq`` (data sequence number)
    columns added by ``execution.py``.

    Reads each position-delete file separately to tag rows with the delete
    file's sequence number, then applies a left-anti join on
    ``(file_path, pos)`` gated by ``data._seq <= delete.seq`` (spec §7.6).
    Paths are normalised on both sides symmetrically (spec §9).
    """
    pos_files = [f for f in delete_files if f.content == 1]
    if not pos_files:
        return data

    log.debug("Applying %d position-delete file(s).", len(pos_files))

    # Read each file individually so _del_seq can be added as a per-file literal.
    del_parts = []
    for f in pos_files:
        path = normalize_path(f.path, normalizer)
        del_parts.append(
            spark.read.parquet(path)
            .select(col("file_path").alias("_del_fp"), col("pos").alias("_del_rid"))
            .withColumn("_del_seq", lit(f.data_sequence_number))
        )

    pos_df = reduce(lambda a, b: a.unionByName(b), del_parts)

    # Normalise paths on both sides so comparisons succeed regardless of which
    # scheme the writer recorded (spec §9).
    if normalizer is not None:
        data = data.withColumn("_fp", _apply_normalizer(col("_fp"), normalizer))
        pos_df = pos_df.withColumn("_del_fp", _apply_normalizer(col("_del_fp"), normalizer))

    return data.join(
        broadcast(pos_df),
        on=(
            (data["_fp"] == pos_df["_del_fp"])
            & (data["_rid"] == pos_df["_del_rid"])
            & (data["_seq"] <= pos_df["_del_seq"])  # spec §7.6 sequence gate
        ),
        how="left_anti",
    )


# ---------------------------------------------------------------------------
# Equality deletes  (spec §7.7)
# ---------------------------------------------------------------------------


def apply_equality_deletes(
    data: DataFrame,
    delete_files: list[DeleteFile],
    spark: SparkSession,
    schema_by_id: dict[int, str],
    partition_specs: dict[int, PartitionSpec],
    normalizer: PathNormalizer | None = None,
) -> DataFrame:
    """Remove rows matched by equality delete files.

    *data* must carry ``_seq`` (data sequence number), ``_spec_id``,
    and ``_partition`` (struct column) added by ``execution.py``.

    Groups delete files by their ``equality_field_ids`` shape.  For each shape,
    reads all delete files, unions them with their seq/partition metadata, then
    applies a sequence-gated (``data._seq < delete.seq``), partition-scoped,
    null-safe anti-join on the equality columns.  Spec §7.7.
    """
    eq_files = [f for f in delete_files if f.content == 2]
    if not eq_files:
        return data

    log.debug("Applying equality deletes from %d file(s).", len(eq_files))

    # Group by equality_field_ids (the "shape")
    shapes: dict[tuple, list[DeleteFile]] = {}
    for df in eq_files:
        shapes.setdefault(df.equality_field_ids, []).append(df)

    for eq_ids, files in shapes.items():
        eq_col_names = []
        for fid in eq_ids:
            name = schema_by_id.get(fid)
            if name is None:
                raise UnsupportedFeatureError(
                    f"Equality delete references field ID {fid} which is not in the current schema."
                )
            eq_col_names.append(name)

        # Union all delete DataFrames of this shape, tagging each with its
        # sequence number and partition info for gating.
        del_parts = []
        for f in files:
            path = normalize_path(f.path, normalizer)
            spec = partition_specs.get(f.spec_id)
            del_df = (
                spark.read.parquet(path)
                .select(*eq_col_names)
                .withColumn("_del_seq", lit(f.data_sequence_number))
                .withColumn("_del_spec_id", lit(f.spec_id))
                .withColumn("_del_partition", _partition_struct_lit(f.partition, spec))
            )
            del_parts.append(del_df)

        all_deletes = reduce(lambda a, b: a.unionByName(b), del_parts)

        # Anti-join condition (spec §7.7):
        #   data._seq < delete.seq  (strict: equality delete applied AFTER data was written)
        #   data._spec_id == delete.spec_id  (same partition spec)
        #   data._partition == delete.partition  (same partition values)
        #   all equality columns null-safe equal (spec §4.7: "Null matches null")
        cond = (
            (data["_seq"] < all_deletes["_del_seq"])
            & (data["_spec_id"] == all_deletes["_del_spec_id"])
            & data["_partition"].eqNullSafe(all_deletes["_del_partition"])  # null-safe struct cmp
        )
        for c in eq_col_names:
            cond = cond & data[c].eqNullSafe(all_deletes[c])

        data = data.join(broadcast(all_deletes), on=cond, how="left_anti")

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_normalizer(column: col, normalizer: PathNormalizer) -> col:
    """Apply a path normaliser to a Spark column.

    For ``DEFAULT_NORMALIZER`` the three built-in remaps are expressed as a
    pure-SQL ``when`` chain (no Python worker needed on executors).  Any other
    normaliser falls back to a Python UDF (requires Python workers; only used
    when a custom normaliser is explicitly provided).
    """
    from pyspark_iceberg_reader.paths import DEFAULT_NORMALIZER

    if normalizer is DEFAULT_NORMALIZER:
        return (
            when(column.startswith("s3://"), regexp_replace(column, "^s3://", "s3a://"))
            .when(column.startswith("wasbs://"), regexp_replace(column, "^wasbs://", "abfss://"))
            .when(column.startswith("dbfs://"), regexp_replace(column, "^dbfs://", "dbfs:/"))
            .otherwise(column)
        )

    # Custom normaliser — use a Python UDF (Python workers required)
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    return udf(normalizer, StringType())(column)


def _partition_struct_lit(partition: tuple, spec: PartitionSpec | None) -> col:
    """Build a Spark struct literal for the given partition values.

    The struct field names come from the partition spec so that both sides of
    the equality-delete join have the same schema, enabling Spark's struct
    ``==`` comparison.  For unpartitioned tables (empty partition), returns
    an empty struct literal.  Spec §4.7 (same-partition scoping).
    """
    if not partition:
        return struct()

    if spec is None or len(spec.fields) != len(partition):
        return struct(*[lit(v).alias(f"_p{i}") for i, v in enumerate(partition)])

    return struct(*[lit(v).alias(f.name) for v, f in zip(partition, spec.fields)])
