"""From-scratch native planner using Spark Avro manifest reads.

Implements the ``ScanPlanner`` protocol using only Spark — no third-party
Iceberg libraries.  Spec §7.3.
"""
from __future__ import annotations

import logging

from pyspark.sql import SparkSession

from pyspark_iceberg_reader.manifests import resolve_manifests
from pyspark_iceberg_reader.metadata import TableMetadata
from pyspark_iceberg_reader.paths import PathNormalizer, resolve_path
from pyspark_iceberg_reader.planning.base import ScanPlan

log = logging.getLogger(__name__)


class NativePlanner:
    """Planner that reads Iceberg manifests via ``spark.read.format("avro")``.

    This is the default planner — it has no dependencies beyond PySpark and
    uses whatever filesystem credentials are already available to the cluster.
    Spec §7.3.
    """

    def __init__(
        self,
        spark: SparkSession,
        normalizer: PathNormalizer | None = None,
    ) -> None:
        self._spark = spark
        self._normalizer = normalizer

    def plan(self, metadata: TableMetadata, snapshot_id: int) -> ScanPlan:
        """Build a ``ScanPlan`` for *snapshot_id* from the Iceberg manifests.

        Reads the manifest list Avro file, then all manifest files, applying
        inheritance rules and filtering live entries.  Spec §7.3.
        """
        snapshot = metadata.snapshots[snapshot_id]
        manifest_list_path = resolve_path(snapshot.manifest_list, metadata.location)

        log.debug(
            "NativePlanner: planning snapshot %d, manifest list %r.",
            snapshot_id,
            manifest_list_path,
        )

        data_files, delete_files = resolve_manifests(
            manifest_list_path,
            self._spark,
            self._normalizer,
        )

        return ScanPlan(
            snapshot_id=snapshot_id,
            current_schema=metadata.current_schema,
            partition_specs=metadata.partition_specs,
            data_files=data_files,
            delete_files=delete_files,
        )
