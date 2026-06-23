"""Tests for metadata.py — parsing and snapshot resolution.

Pure-Python tests (no ``spark`` fixture) cover all parsing logic.
The single integration test at the bottom verifies Spark can load a local file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from pyspark_iceberg_reader.errors import (
    MetadataParseError,
    SnapshotNotFoundError,
    UnsupportedFormatVersionError,
)
from pyspark_iceberg_reader.metadata import (
    IcebergField,
    IcebergSchema,
    PartitionSpec,
    Snapshot,
    TableMetadata,
    _parse_metadata_json,
    read_metadata,
    resolve_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Snapshot ID used by v2_single_snapshot.metadata.json
_SNAPSHOT_ID = 3776207205136740864

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> str:
    """Return the raw JSON text of a fixture file."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def _minimal_meta(current_snapshot_id: int | None = _SNAPSHOT_ID) -> TableMetadata:
    """Build a minimal TableMetadata for resolve_snapshot tests."""
    schema = IcebergSchema(
        schema_id=0,
        fields=(IcebergField(field_id=1, name="id", field_type="long", required=True),),
    )
    snapshot = Snapshot(
        snapshot_id=_SNAPSHOT_ID,
        sequence_number=1,
        manifest_list="s3://bucket/snap.avro",
        schema_id=0,
    )
    return TableMetadata(
        table_uuid="test-uuid",
        location="s3://bucket/table",
        current_snapshot_id=current_snapshot_id,
        current_schema_id=0,
        default_spec_id=0,
        schemas={0: schema},
        partition_specs={0: PartitionSpec(spec_id=0, fields=())},
        snapshots={_SNAPSHOT_ID: snapshot},
    )


# ---------------------------------------------------------------------------
# _parse_metadata_json — pure Python, no Spark
# ---------------------------------------------------------------------------


def test_parse_valid_v2_returns_table_metadata():
    meta = _parse_metadata_json(_load_fixture("v2_single_snapshot.metadata.json"))

    assert meta.table_uuid == "9c12d441-03fe-4693-9a96-a0705ddf69c1"
    assert meta.location == "s3://test-bucket/warehouse/db/events"
    assert meta.current_schema_id == 0
    assert meta.default_spec_id == 0
    assert meta.current_snapshot_id == _SNAPSHOT_ID


def test_parse_schema_fields():
    meta = _parse_metadata_json(_load_fixture("v2_single_snapshot.metadata.json"))
    fields = meta.current_schema.fields

    assert len(fields) == 3
    assert fields[0].field_id == 1
    assert fields[0].name == "id"
    assert fields[0].field_type == "long"
    assert fields[0].required is True
    assert fields[1].required is False


def test_parse_partition_spec():
    meta = _parse_metadata_json(_load_fixture("v2_single_snapshot.metadata.json"))
    spec = meta.partition_specs[0]

    assert len(spec.fields) == 1
    assert spec.fields[0].source_id == 3
    assert spec.fields[0].transform == "identity"
    assert spec.fields[0].field_id == 1000


def test_parse_snapshots():
    meta = _parse_metadata_json(_load_fixture("v2_single_snapshot.metadata.json"))
    snap = meta.snapshots[_SNAPSHOT_ID]

    assert snap.sequence_number == 1
    assert snap.schema_id == 0
    assert "snap-" in snap.manifest_list


def test_parse_empty_table_has_no_current_snapshot():
    meta = _parse_metadata_json(_load_fixture("v2_empty_table.metadata.json"))

    assert meta.current_snapshot_id is None
    assert meta.snapshots == {}


def test_parse_rejects_v1():
    v1 = json.dumps({"format-version": 1})
    with pytest.raises(UnsupportedFormatVersionError) as exc_info:
        _parse_metadata_json(v1)
    assert exc_info.value.version == 1


def test_parse_rejects_v3():
    v3 = json.dumps({"format-version": 3})
    with pytest.raises(UnsupportedFormatVersionError) as exc_info:
        _parse_metadata_json(v3)
    assert exc_info.value.version == 3


def test_parse_raises_on_missing_format_version():
    with pytest.raises(MetadataParseError, match="format-version"):
        _parse_metadata_json(json.dumps({"table-uuid": "x"}))


def test_parse_raises_on_missing_required_field():
    # format-version present and valid, but other required fields absent
    with pytest.raises(MetadataParseError):
        _parse_metadata_json(json.dumps({"format-version": 2}))


def test_parse_raises_on_invalid_json():
    with pytest.raises(MetadataParseError, match="Invalid JSON"):
        _parse_metadata_json("{not: valid json")


def test_parse_normalises_minus1_sentinel_to_none():
    """Some Iceberg writers emit -1 for current-snapshot-id on an empty table."""
    raw = json.loads(_load_fixture("v2_empty_table.metadata.json"))
    raw["current-snapshot-id"] = -1
    meta = _parse_metadata_json(json.dumps(raw))
    assert meta.current_snapshot_id is None


# ---------------------------------------------------------------------------
# resolve_snapshot — pure Python, no Spark
# ---------------------------------------------------------------------------


def test_resolve_uses_current_snapshot_by_default():
    meta = _minimal_meta()
    snap = resolve_snapshot(meta, snapshot_id=None)
    assert snap is not None
    assert snap.snapshot_id == _SNAPSHOT_ID


def test_resolve_explicit_snapshot_id():
    meta = _minimal_meta()
    snap = resolve_snapshot(meta, snapshot_id=_SNAPSHOT_ID)
    assert snap.snapshot_id == _SNAPSHOT_ID


def test_resolve_unknown_snapshot_raises():
    meta = _minimal_meta()
    with pytest.raises(SnapshotNotFoundError) as exc_info:
        resolve_snapshot(meta, snapshot_id=99999)
    assert exc_info.value.snapshot_id == 99999


def test_resolve_empty_table_returns_none():
    """An empty table (current-snapshot-id=None) should return None."""
    meta = _minimal_meta(current_snapshot_id=None)
    assert resolve_snapshot(meta, snapshot_id=None) is None


def test_resolve_explicit_id_on_empty_table_raises():
    """Requesting a snapshot by ID on an empty table must raise, not return None."""
    meta = TableMetadata(
        table_uuid="x",
        location="s3://b/t",
        current_snapshot_id=None,
        current_schema_id=0,
        default_spec_id=0,
        schemas={0: IcebergSchema(schema_id=0, fields=())},
        partition_specs={0: PartitionSpec(spec_id=0, fields=())},
        snapshots={},
    )
    with pytest.raises(SnapshotNotFoundError):
        resolve_snapshot(meta, snapshot_id=_SNAPSHOT_ID)


def test_resolve_current_snapshot_id_missing_from_snapshots_raises():
    """current-snapshot-id set but absent from snapshots dict raises SnapshotNotFoundError.

    This covers the gap where reader.py would previously pass the orphan ID to
    the planner, causing a raw KeyError instead of a typed exception.
    """
    meta = TableMetadata(
        table_uuid="x",
        location="s3://b/t",
        current_snapshot_id=99,
        current_schema_id=0,
        default_spec_id=0,
        schemas={0: IcebergSchema(schema_id=0, fields=())},
        partition_specs={0: PartitionSpec(spec_id=0, fields=())},
        snapshots={},  # snapshot 99 is NOT present
    )
    with pytest.raises(SnapshotNotFoundError) as exc_info:
        resolve_snapshot(meta, snapshot_id=None)
    assert exc_info.value.snapshot_id == 99


# ---------------------------------------------------------------------------
# read_metadata — integration test (requires Spark)
# ---------------------------------------------------------------------------


def test_read_metadata_via_spark(spark: SparkSession):
    """Verify Spark can load a local metadata file end-to-end."""
    path = (FIXTURES / "v2_single_snapshot.metadata.json").as_uri()
    meta = read_metadata(path, spark)

    assert meta.table_uuid == "9c12d441-03fe-4693-9a96-a0705ddf69c1"
    assert meta.current_snapshot_id == _SNAPSHOT_ID
    assert 0 in meta.schemas
    assert 0 in meta.partition_specs
    assert meta.metadata_location == path
