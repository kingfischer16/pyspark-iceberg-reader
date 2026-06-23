"""Shared intermediate representation (IR) produced by NativePlanner.

Spec §5.1 (ScanPlan IR), §5.2 (ScanPlanner protocol).

``NativePlanner`` produces a ``ScanPlan``; ``execution.py`` consumes it.
All coupling between planner and executor is through these dataclasses and
the ``ScanPlanner`` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyspark_iceberg_reader.metadata import IcebergSchema, PartitionSpec, TableMetadata


@dataclass(frozen=True)
class DataFileTask:
    """A single live data file to read.

    Spec §5.1.
    """

    path: str
    file_format: str  # "PARQUET" — others rejected at planning time
    spec_id: int
    partition: tuple  # partition values as a tuple, spec_id-qualified
    data_sequence_number: int
    record_count: int


@dataclass(frozen=True)
class DeleteFile:
    """A single live delete file to apply.

    Spec §5.1.  ``content`` mirrors the Iceberg ``data_file.content`` field:
    ``1`` = position deletes, ``2`` = equality deletes.
    """

    path: str
    file_format: str
    content: int  # 1 = position, 2 = equality
    spec_id: int
    partition: tuple
    data_sequence_number: int
    equality_field_ids: tuple  # empty tuple for position-delete files


@dataclass(frozen=True)
class ScanPlan:
    """Complete plan for reading a snapshot.

    The driver-side footprint is metadata-scale (file lists + small schema objects).
    All bulk data reads happen inside ``execution.py`` via Spark.  Spec §5.1.
    """

    snapshot_id: int
    current_schema: IcebergSchema
    partition_specs: dict[int, PartitionSpec]
    data_files: list[DataFileTask]
    delete_files: list[DeleteFile]


class ScanPlanner(Protocol):
    """Interface implemented by ``NativePlanner``.

    Spec §5.2.
    """

    def plan(self, metadata: TableMetadata, snapshot_id: int) -> ScanPlan:
        """Build a ``ScanPlan`` from the resolved snapshot."""
        ...
