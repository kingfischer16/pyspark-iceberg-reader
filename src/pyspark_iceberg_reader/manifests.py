"""Manifest list / manifest file reading and Iceberg inheritance rules.

Spec §4.3 (manifest list), §4.4 (manifest entries), §4.5 (inheritance),
§4.6 (live files), §7.3 (native planning algorithm).

Design split (for testability without cluster):
- File I/O:  ``read_manifest_list`` and ``resolve_manifests`` read Avro via Spark.
             These require ``spark.read.format("avro")`` → cluster-only tests.
- Logic:     ``_build_scan_items`` accepts pre-collected Row objects.  Tests
             construct Row objects directly in pure Python — no Spark needed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from pyspark_iceberg_reader.errors import UnsupportedFeatureError
from pyspark_iceberg_reader.paths import PathNormalizer, normalize_path
from pyspark_iceberg_reader.planning.base import DataFileTask, DeleteFile

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Manifest-entry ``status`` field values (spec §4.4)
_STATUS_EXISTING = 0
_STATUS_ADDED = 1
_STATUS_DELETED = 2

# data_file ``content`` field values (spec §4.4)
_CONTENT_DATA = 0
_CONTENT_POSITION_DELETES = 1
_CONTENT_EQUALITY_DELETES = 2


# ---------------------------------------------------------------------------
# File I/O — require spark.read.format("avro"); cluster-only end-to-end tests
# ---------------------------------------------------------------------------


def read_manifest_list(manifest_list_path: str, spark: SparkSession) -> DataFrame:
    """Read a manifest-list Avro file → raw DataFrame.

    Each row describes one manifest file in the snapshot.  Spec §4.3.
    Column names exactly match the Iceberg Avro spec fields:
    ``manifest_path``, ``content``, ``sequence_number``, ``added_snapshot_id``,
    ``partition_spec_id``.
    """
    return spark.read.format("avro").load(manifest_list_path)


def resolve_manifests(
    manifest_list_path: str,
    spark: SparkSession,
    normalizer: PathNormalizer | None = None,
) -> tuple[list[DataFileTask], list[DeleteFile]]:
    """Full pipeline: read manifest list → read manifest files → build IR.

    Applies inheritance (spec §4.5), drops DELETED tombstones (spec §4.6),
    and rejects non-Parquet file formats with ``UnsupportedFeatureError``.
    Spec §7.3.
    """
    ml_df = read_manifest_list(manifest_list_path, spark)

    ml_rows = ml_df.select(
        "manifest_path",
        "content",
        "sequence_number",
        "added_snapshot_id",
        "partition_spec_id",
    ).collect()

    if not ml_rows:
        return [], []

    # Normalize manifest-list keys so they can be matched against
    # _metadata.file_path values that may use a different scheme (spec §9).
    ml_meta = {normalize_path(r.manifest_path, normalizer): r for r in ml_rows}
    all_paths = [r.manifest_path for r in ml_rows]

    log.debug("Reading %d manifest file(s).", len(all_paths))
    raw = (
        spark.read.format("avro")
        .load(all_paths)
        .withColumn("_manifest_path", col("_metadata.file_path"))
    )

    # Project to only the columns we need before collecting to driver (§8 — minimize footprint)
    projected_rows = raw.select(
        "status",
        "snapshot_id",
        "sequence_number",
        col("data_file.content").alias("df_content"),
        col("data_file.file_path").alias("df_file_path"),
        col("data_file.file_format").alias("df_file_format"),
        col("data_file.partition").alias("df_partition"),
        col("data_file.record_count").alias("df_record_count"),
        col("data_file.equality_ids").alias("df_equality_ids"),
        "_manifest_path",
    ).collect()

    return _build_scan_items(projected_rows, ml_meta, normalizer)


# ---------------------------------------------------------------------------
# Core logic — works with any DataFrame that has the right schema
# ---------------------------------------------------------------------------


def _build_scan_items(
    entries: list,
    ml_meta: dict,  # manifest_path -> Row(sequence_number, added_snapshot_id, partition_spec_id, content)
    normalizer: PathNormalizer | None,
) -> tuple[list[DataFileTask], list[DeleteFile]]:
    """Apply inheritance, filter live entries, return ``(data_files, delete_files)``.

    *entries* is a list of already-collected Row objects with fields:
    ``status``, ``snapshot_id``, ``sequence_number``, ``df_content``,
    ``df_file_path``, ``df_file_format``, ``df_partition``,
    ``df_record_count``, ``df_equality_ids``, ``_manifest_path``.

    The caller projects and collects the Avro DataFrame before calling this;
    logic is pure Python so tests can pass manually constructed Row objects
    without needing ``spark.createDataFrame``.  Spec §4.5, §4.6.
    """
    rows = entries

    data_files: list[DataFileTask] = []
    delete_files: list[DeleteFile] = []

    for row in rows:
        # spec §4.6 — drop DELETED tombstones first; no further processing needed
        if row.status == _STATUS_DELETED:
            continue

        # Normalize path before lookup so scheme mismatches (s3 vs s3a) don't
        # silently lose entries (spec §9).
        ml_row = ml_meta.get(normalize_path(row._manifest_path, normalizer))
        if ml_row is None:
            log.warning("Manifest path %r not found in manifest list; skipping.", row._manifest_path)
            continue

        # spec §4.5 — inheritance
        seq, _snap = _resolve_inheritance(
            status=row.status,
            entry_seq=row.sequence_number,
            entry_snapshot_id=row.snapshot_id,
            manifest_seq=ml_row.sequence_number,
            manifest_snapshot_id=ml_row.added_snapshot_id,
        )

        path = normalize_path(row.df_file_path, normalizer)

        if row.df_file_format.upper() != "PARQUET":
            raise UnsupportedFeatureError(
                f"Non-Parquet data files are not supported: format={row.df_file_format!r}, path={path!r}"
            )

        partition = tuple(row.df_partition) if row.df_partition is not None else ()
        spec_id: int = ml_row.partition_spec_id

        if row.df_content == _CONTENT_DATA:
            if row.df_record_count is None:
                raise UnsupportedFeatureError(
                    f"Manifest entry for {path!r} has null record_count — corrupt manifest."
                )
            data_files.append(
                DataFileTask(
                    path=path,
                    file_format=row.df_file_format.upper(),
                    spec_id=spec_id,
                    partition=partition,
                    data_sequence_number=seq,
                    record_count=int(row.df_record_count),
                )
            )
        else:
            # _CONTENT_POSITION_DELETES or _CONTENT_EQUALITY_DELETES
            delete_files.append(
                DeleteFile(
                    path=path,
                    file_format=row.df_file_format.upper(),
                    content=int(row.df_content),
                    spec_id=spec_id,
                    partition=partition,
                    data_sequence_number=seq,
                    equality_field_ids=tuple(row.df_equality_ids or []),
                )
            )

    log.debug(
        "Resolved %d data file(s) and %d delete file(s).",
        len(data_files),
        len(delete_files),
    )
    return data_files, delete_files


# ---------------------------------------------------------------------------
# Inheritance helper — pure Python, no Spark
# ---------------------------------------------------------------------------


def _resolve_inheritance(
    status: int,
    entry_seq: int | None,
    entry_snapshot_id: int | None,
    manifest_seq: int | None,
    manifest_snapshot_id: int | None,
) -> tuple[int, int | None]:
    """Resolve sequence number and snapshot id for one manifest entry.

    Spec §4.5:
    - ADDED (status=1) with null values → inherit from the owning manifest.
    - EXISTING (status=0) or DELETED (status=2) → values must be explicit.

    Returns ``(data_sequence_number, snapshot_id)``.
    Raises ``UnsupportedFeatureError`` if a required value is missing.
    """
    if status == _STATUS_ADDED:
        seq = entry_seq if entry_seq is not None else manifest_seq
        snap = entry_snapshot_id if entry_snapshot_id is not None else manifest_snapshot_id
        if seq is None:
            raise UnsupportedFeatureError(
                "ADDED manifest entry has null sequence_number and the owning manifest "
                "also has null sequence_number — corrupt manifest."
            )
        return int(seq), snap
    else:
        # EXISTING or DELETED: must be explicit
        if entry_seq is None:
            raise UnsupportedFeatureError(
                f"Manifest entry with status={status} (EXISTING/DELETED) has null "
                "sequence_number — corrupt manifest."
            )
        return int(entry_seq), entry_snapshot_id
