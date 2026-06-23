"""Top-level entry point: ``read_iceberg_table``.

Orchestrates metadata read → snapshot resolution → planning → execution.
Spec §7.8.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession

from pyspark_iceberg_reader.execution import _empty_dataframe, execute_scan_plan
from pyspark_iceberg_reader.metadata import read_metadata, resolve_snapshot
from pyspark_iceberg_reader.planning.native import NativePlanner
from pyspark_iceberg_reader.schema import iceberg_schema_to_spark

if TYPE_CHECKING:
    from pyspark_iceberg_reader.paths import PathNormalizer

log = logging.getLogger(__name__)


def read_iceberg_table(
    metadata_location: str,
    *,
    snapshot_id: int | None = None,
    normalizer: PathNormalizer | None = None,
    spark: SparkSession | None = None,
) -> DataFrame:
    """Read an Iceberg v2 table into a Spark ``DataFrame``.

    :param metadata_location: Absolute path to a ``*.metadata.json`` file
        (or ``.metadata.json.gz``).  Must be readable by Spark (no credentials
        embedded here — use the cluster's existing auth).
    :param snapshot_id: Snapshot to read. Defaults to the table's
        ``current-snapshot-id``.  Must exist or raises
        :class:`~pyspark_iceberg_reader.errors.SnapshotNotFoundError`.
        **Never** inferred from timestamps or "latest" heuristics.
    :param normalizer: Optional path-scheme normaliser.  Use
        :data:`~pyspark_iceberg_reader.paths.DEFAULT_NORMALIZER` for standard
        cloud scheme remapping (``s3`` → ``s3a``, etc.).
    :param spark: SparkSession to use.  Falls back to the active session, then
        creates a new one.
    :returns: ``DataFrame`` with the table's current schema.  Deletes are
        applied; partition columns are *not* materialised (hidden partitioning).
    :raises UnsupportedFormatVersionError: If the table is not Iceberg v2.
    :raises SnapshotNotFoundError: If *snapshot_id* does not exist.
    :raises UnsupportedFeatureError: For ORC files, corrupt manifests, etc.

    Spec §7.8.
    """
    if spark is None:
        spark = SparkSession.getActiveSession()
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    log.debug("read_iceberg_table: metadata=%r", metadata_location)

    metadata = read_metadata(metadata_location, spark)

    # Resolve snapshot (spec §7.2 — never infer, never guess).
    # resolve_snapshot raises SnapshotNotFoundError for missing explicit IDs
    # and for a current-snapshot-id that is absent from the snapshots map.
    snapshot = resolve_snapshot(metadata, snapshot_id)

    if snapshot is None:
        # Empty table: no snapshots yet
        log.debug("read_iceberg_table: table has no snapshots, returning empty DataFrame.")
        return _empty_dataframe(iceberg_schema_to_spark(metadata.current_schema), spark)

    plan = NativePlanner(spark, normalizer).plan(metadata, snapshot.snapshot_id)
    return execute_scan_plan(plan, spark, normalizer)
