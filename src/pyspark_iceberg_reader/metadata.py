"""Parse Iceberg v2 ``*.metadata.json`` files and resolve snapshots.

Spec §4.2 (metadata fields), §7.1 (reading via Spark), §7.2 (snapshot resolution).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from pyspark.sql import SparkSession

from pyspark_iceberg_reader.errors import (
    MetadataParseError,
    SnapshotNotFoundError,
    UnsupportedFormatVersionError,
)

log = logging.getLogger(__name__)

_SUPPORTED_VERSION = 2
# Some Iceberg writers emit -1 rather than null/omit for "no current snapshot"
_EMPTY_SNAPSHOT_SENTINEL = -1


# ---------------------------------------------------------------------------
# Domain types  (consumed by schema.py, planning/base.py, and execution.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IcebergField:
    """A single field in an Iceberg schema.

    :param field_id: Stable numeric ID that survives column renames. This is
        the key used for field-ID parquet reads in schema.py (spec §4.8).
    :param name: Column name in the *current* schema.
    :param field_type: Raw Iceberg type from the JSON — either a primitive
        string (e.g. ``"long"``, ``"string"``) or a complex-type dict
        (``{"type": "struct", "fields": [...]}``, ``{"type": "list", ...}``,
        ``{"type": "map", ...}``). schema.py converts this to a Spark DataType.
    :param required: ``True`` if the column is non-nullable.
    :param doc: Optional column-level documentation string from the schema.
    """

    field_id: int
    name: str
    field_type: str | dict[str, Any]
    required: bool
    doc: str | None = None


@dataclass(frozen=True)
class IcebergSchema:
    """An Iceberg schema, identified by ``schema-id``.

    :param schema_id: Matches ``schemas[].schema-id`` in metadata.json.
    :param fields: Top-level fields. Nested type trees are stored raw inside
        each :attr:`IcebergField.field_type`; schema.py recurses into them.
    """

    schema_id: int
    fields: tuple[IcebergField, ...]


@dataclass(frozen=True)
class PartitionField:
    """A single field in a partition spec — a transform applied to a source column.

    :param source_id: Field ID of the column being partitioned.
    :param field_id: Partition field ID used when scoping delete files (spec §4.7).
    :param name: Partition field name (derived column, not in the output DataFrame).
    :param transform: Transform string, e.g. ``"identity"``, ``"bucket[16]"``, ``"day"``.
    """

    source_id: int
    field_id: int
    name: str
    transform: str


@dataclass(frozen=True)
class PartitionSpec:
    """Partition specification, identified by ``spec-id``.

    :param spec_id: Matches ``partition-specs[].spec-id`` in metadata.json.
    :param fields: Partition fields in declaration order.
    """

    spec_id: int
    fields: tuple[PartitionField, ...]


@dataclass(frozen=True)
class Snapshot:
    """A single table snapshot.

    :param snapshot_id: Unique numeric snapshot ID.
    :param sequence_number: Data sequence number for this snapshot (spec §4.5).
        Used to determine delete applicability — do not confuse with file-level
        sequence numbers stored in manifest entries.
    :param manifest_list: Absolute path to the manifest list Avro file.
    :param schema_id: Schema in effect when this snapshot was written. Note: the
        output DataFrame always uses the *table's* ``current-schema-id``, not
        this value (spec §7.2).
    """

    snapshot_id: int
    sequence_number: int
    manifest_list: str
    schema_id: int | None


@dataclass(frozen=True)
class TableMetadata:
    """Parsed contents of a ``*.metadata.json`` file.

    :param table_uuid: Stable UUID for the table.
    :param location: Root location; used to resolve relative paths in manifests.
    :param current_snapshot_id: The table's current snapshot ID, or ``None``
        for an empty table (no snapshots written yet).
    :param current_schema_id: ID of the schema that defines the output DataFrame
        columns. Always use this — not snapshot ``schema_id`` — for the output.
    :param default_spec_id: Partition spec applied to newly written files.
    :param schemas: All known schemas keyed by ``schema-id``.
    :param partition_specs: All partition specs keyed by ``spec-id``.
    :param snapshots: All snapshots keyed by ``snapshot-id``.
    :param metadata_location: Absolute path to the ``*.metadata.json`` file that
        was read.  Empty string when constructed via ``_parse_metadata_json``
        directly (tests); always populated by ``read_metadata``.
    """

    table_uuid: str
    location: str
    current_snapshot_id: int | None
    current_schema_id: int
    default_spec_id: int
    schemas: dict[int, IcebergSchema]
    partition_specs: dict[int, PartitionSpec]
    snapshots: dict[int, Snapshot]
    metadata_location: str = ""

    @property
    def current_schema(self) -> IcebergSchema:
        """The schema used to define the output DataFrame columns."""
        return self.schemas[self.current_schema_id]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read_metadata(metadata_location: str, spark: SparkSession) -> TableMetadata:
    """Read and parse a ``*.metadata.json`` (or ``.gz``) from any Spark-accessible path.

    Uses :meth:`~pyspark.sql.DataFrameReader.text` so that the cluster's existing
    filesystem credentials are used transparently — no cloud SDK calls. Spec §7.1.

    :param metadata_location: Full path to the metadata file, e.g.
        ``s3://bucket/table/metadata/00001-abc.metadata.json``.
    :param spark: Active ``SparkSession``.
    :returns: Parsed :class:`TableMetadata`.
    :raises UnsupportedFormatVersionError: If ``format-version`` is not 2.
    :raises MetadataParseError: If the file is malformed or missing required fields.
    """
    log.debug("Reading metadata from %s", metadata_location)
    text = spark.read.text(metadata_location, wholetext=True).first()[0]
    meta = _parse_metadata_json(text)
    return replace(meta, metadata_location=metadata_location)


def resolve_snapshot(meta: TableMetadata, snapshot_id: int | None) -> Snapshot | None:
    """Resolve which snapshot to scan. Returns ``None`` for an empty table.

    Never guesses — uses ``current-snapshot-id`` when ``snapshot_id`` is ``None``,
    or validates the caller-supplied ID against the known snapshot list. Spec §7.2.

    :param meta: Parsed table metadata.
    :param snapshot_id: Caller-requested snapshot ID, or ``None`` to use the
        table's current snapshot.
    :returns: The resolved :class:`Snapshot`, or ``None`` if the table has no
        snapshots (caller should return an empty DataFrame).
    :raises SnapshotNotFoundError: If ``snapshot_id`` is provided but does not
        exist in ``meta.snapshots``.
    """
    if snapshot_id is None:
        if meta.current_snapshot_id is None:
            return None  # empty table; caller should return an empty DataFrame
        target_id = meta.current_snapshot_id
    else:
        target_id = snapshot_id

    if target_id not in meta.snapshots:
        raise SnapshotNotFoundError(target_id)

    return meta.snapshots[target_id]


# ---------------------------------------------------------------------------
# Internal helpers  (public only for unit testing; not part of the package API)
# ---------------------------------------------------------------------------


def _parse_metadata_json(text: str) -> TableMetadata:
    """Parse a metadata JSON string into a :class:`TableMetadata`.

    Separated from :func:`read_metadata` so that unit tests can exercise parsing
    logic without a SparkSession. Spec §4.2, §7.1.

    :param text: Raw contents of a ``*.metadata.json`` file.
    :raises MetadataParseError: On invalid JSON or any missing required field.
    :raises UnsupportedFormatVersionError: When ``format-version`` is not 2.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetadataParseError(f"Invalid JSON: {exc}") from exc

    try:
        version = raw["format-version"]
    except KeyError as exc:
        raise MetadataParseError("Missing required field 'format-version'") from exc

    if version != _SUPPORTED_VERSION:
        raise UnsupportedFormatVersionError(version)

    try:
        schemas = {s["schema-id"]: _parse_schema(s) for s in raw["schemas"]}
        specs = {p["spec-id"]: _parse_partition_spec(p) for p in raw["partition-specs"]}
        snapshots = {s["snapshot-id"]: _parse_snapshot(s) for s in raw.get("snapshots", [])}

        raw_current = raw.get("current-snapshot-id")
        # Normalise -1 sentinel (used by some Iceberg writers) to None
        current_snapshot_id = (
            None if (raw_current is None or raw_current == _EMPTY_SNAPSHOT_SENTINEL)
            else raw_current
        )

        return TableMetadata(
            table_uuid=raw["table-uuid"],
            location=raw["location"],
            current_snapshot_id=current_snapshot_id,
            current_schema_id=raw["current-schema-id"],
            default_spec_id=raw["default-spec-id"],
            schemas=schemas,
            partition_specs=specs,
            snapshots=snapshots,
        )
    except KeyError as exc:
        raise MetadataParseError(f"Missing required metadata field: {exc}") from exc


def _parse_schema(raw: dict[str, Any]) -> IcebergSchema:
    fields = tuple(
        IcebergField(
            field_id=f["id"],
            name=f["name"],
            field_type=f["type"],  # kept raw; schema.py converts to Spark DataType
            required=f.get("required", False),
            doc=f.get("doc"),
        )
        for f in raw["fields"]
    )
    return IcebergSchema(schema_id=raw["schema-id"], fields=fields)


def _parse_partition_spec(raw: dict[str, Any]) -> PartitionSpec:
    fields = tuple(
        PartitionField(
            source_id=f["source-id"],
            field_id=f["field-id"],
            name=f["name"],
            transform=f["transform"],
        )
        for f in raw["fields"]
    )
    return PartitionSpec(spec_id=raw["spec-id"], fields=fields)


def _parse_snapshot(raw: dict[str, Any]) -> Snapshot:
    return Snapshot(
        snapshot_id=raw["snapshot-id"],
        sequence_number=raw.get("sequence-number", 0),
        manifest_list=raw["manifest-list"],
        schema_id=raw.get("schema-id"),
    )
