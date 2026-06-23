"""Tests for manifests.py — inheritance, status filtering, and IR shape.

``_build_scan_items`` accepts a list of Row objects (already collected from a
Spark DataFrame), so all tests here are pure Python — no Spark session, no
file I/O, no cluster needed.

The full end-to-end pipeline (reading actual Avro files from disk, then
reading the data files) is covered by ``test_runtime_capabilities.py``
which is ``@pytest.mark.cluster``.
"""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from pyspark_iceberg_reader.errors import UnsupportedFeatureError
from pyspark_iceberg_reader.manifests import _build_scan_items, _resolve_inheritance

_MANIFEST_A = "s3://test-bucket/metadata/manifest_a.avro"
_MANIFEST_B = "s3://test-bucket/metadata/manifest_b.avro"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    status: int,
    seq: int | None,
    snap_id: int | None,
    content: int,
    path: str,
    manifest_path: str = _MANIFEST_A,
    record_count: int = 10,
    equality_ids: list[int] | None = None,
) -> Row:
    """Build a Row mimicking a projected manifest entry (post-DataFrame-collect)."""
    return Row(
        status=status,
        snapshot_id=snap_id,
        sequence_number=seq,
        df_content=content,
        df_file_path=path,
        df_file_format="PARQUET",
        df_partition=Row(part_0="X"),
        df_record_count=record_count,
        df_equality_ids=equality_ids or [],
        _manifest_path=manifest_path,
    )


def _ml_row(
    manifest_path: str = _MANIFEST_A,
    seq: int = 10,
    snap_id: int = 99,
    spec_id: int = 0,
    content: int = 0,
) -> dict:
    return {
        manifest_path: Row(
            sequence_number=seq,
            added_snapshot_id=snap_id,
            partition_spec_id=spec_id,
            content=content,
        )
    }


# ---------------------------------------------------------------------------
# _resolve_inheritance — pure Python
# ---------------------------------------------------------------------------


def test_resolve_inheritance_added_null_seq_inherits():
    seq, _ = _resolve_inheritance(
        status=1, entry_seq=None, entry_snapshot_id=None,
        manifest_seq=10, manifest_snapshot_id=99,
    )
    assert seq == 10


def test_resolve_inheritance_added_explicit_seq_kept():
    seq, snap = _resolve_inheritance(
        status=1, entry_seq=5, entry_snapshot_id=42,
        manifest_seq=10, manifest_snapshot_id=99,
    )
    assert seq == 5
    assert snap == 42


def test_resolve_inheritance_existing_explicit_seq_kept():
    seq, _ = _resolve_inheritance(
        status=0, entry_seq=3, entry_snapshot_id=30,
        manifest_seq=10, manifest_snapshot_id=99,
    )
    assert seq == 3


def test_resolve_inheritance_existing_null_seq_raises():
    with pytest.raises(UnsupportedFeatureError):
        _resolve_inheritance(
            status=0, entry_seq=None, entry_snapshot_id=None,
            manifest_seq=10, manifest_snapshot_id=99,
        )


def test_resolve_inheritance_added_null_manifest_seq_raises():
    with pytest.raises(UnsupportedFeatureError):
        _resolve_inheritance(
            status=1, entry_seq=None, entry_snapshot_id=None,
            manifest_seq=None, manifest_snapshot_id=None,
        )


# ---------------------------------------------------------------------------
# _build_scan_items — pure Python (Row objects, no Spark session)
# ---------------------------------------------------------------------------


def test_build_scan_items_reads_data_file():
    rows = [_entry(status=1, seq=None, snap_id=None, content=0, path="s3://b/data/f1.parquet")]
    meta = _ml_row(seq=10)

    data_files, delete_files = _build_scan_items(rows, meta, None)

    assert len(data_files) == 1
    assert len(delete_files) == 0
    f = data_files[0]
    assert f.path == "s3://b/data/f1.parquet"
    assert f.data_sequence_number == 10  # inherited from manifest list
    assert f.spec_id == 0
    assert f.record_count == 10


def test_build_scan_items_reads_position_delete():
    rows = [_entry(status=1, seq=20, snap_id=99, content=1,
                   path="s3://b/deletes/pos.parquet",
                   manifest_path=_MANIFEST_B)]
    meta = _ml_row(_MANIFEST_B, seq=20, content=1)

    data_files, delete_files = _build_scan_items(rows, meta, None)

    assert len(data_files) == 0
    assert len(delete_files) == 1
    d = delete_files[0]
    assert d.content == 1  # position delete
    assert d.equality_field_ids == ()


def test_build_scan_items_reads_equality_delete():
    rows = [_entry(status=1, seq=15, snap_id=88, content=2,
                   path="s3://b/deletes/eq.parquet",
                   equality_ids=[1, 2])]
    meta = _ml_row(seq=15)

    data_files, delete_files = _build_scan_items(rows, meta, None)

    assert len(delete_files) == 1
    assert delete_files[0].content == 2  # equality delete
    assert delete_files[0].equality_field_ids == (1, 2)


def test_build_scan_items_inheritance_fills_null_seq_for_added():
    rows = [
        # null seq — should inherit manifest's seq=10
        _entry(status=1, seq=None, snap_id=None, content=0, path="s3://b/data/inherit.parquet"),
        # explicit seq — should NOT be overridden
        _entry(status=1, seq=5, snap_id=42, content=0, path="s3://b/data/explicit.parquet"),
    ]
    meta = _ml_row(seq=10)

    data_files, _ = _build_scan_items(rows, meta, None)

    seqs = {f.path: f.data_sequence_number for f in data_files}
    assert seqs["s3://b/data/inherit.parquet"] == 10
    assert seqs["s3://b/data/explicit.parquet"] == 5


def test_build_scan_items_existing_null_seq_raises():
    rows = [_entry(status=0, seq=None, snap_id=None, content=0, path="s3://b/data/bad.parquet")]
    meta = _ml_row()

    with pytest.raises(UnsupportedFeatureError):
        _build_scan_items(rows, meta, None)


def test_build_scan_items_drops_deleted_status():
    rows = [
        _entry(status=2, seq=1, snap_id=10, content=0, path="s3://b/data/tombstone.parquet"),
        _entry(status=1, seq=5, snap_id=50, content=0, path="s3://b/data/live.parquet"),
    ]
    meta = _ml_row()

    data_files, _ = _build_scan_items(rows, meta, None)

    assert len(data_files) == 1
    assert data_files[0].path == "s3://b/data/live.parquet"


def test_build_scan_items_rejects_non_parquet():
    orc_row = Row(
        status=1, snapshot_id=1, sequence_number=1,
        df_content=0, df_file_path="s3://b/data/orc.orc",
        df_file_format="ORC",
        df_partition=Row(part_0="X"),
        df_record_count=5,
        df_equality_ids=[],
        _manifest_path=_MANIFEST_A,
    )
    meta = _ml_row()

    with pytest.raises(UnsupportedFeatureError):
        _build_scan_items([orc_row], meta, None)


def test_build_scan_items_mixed_data_and_deletes():
    rows = [
        _entry(status=1, seq=5, snap_id=1, content=0, path="s3://b/data/d1.parquet"),
        _entry(status=1, seq=8, snap_id=1, content=1, path="s3://b/del/pd.parquet",
               manifest_path=_MANIFEST_B),
        _entry(status=0, seq=3, snap_id=1, content=0, path="s3://b/data/d2.parquet"),
    ]
    meta = {**_ml_row(_MANIFEST_A, seq=5, content=0), **_ml_row(_MANIFEST_B, seq=8, content=1)}

    data_files, delete_files = _build_scan_items(rows, meta, None)

    assert len(data_files) == 2
    assert len(delete_files) == 1


def test_deleted_entry_with_orc_format_is_silently_dropped():
    """DELETED tombstones are skipped before format validation.

    A DELETED entry with format=ORC and null sequence_number must be silently
    dropped — not raise UnsupportedFeatureError — because DELETED entries are
    tombstones we discard before any further processing.
    """
    row = Row(
        status=2,  # DELETED
        snapshot_id=None,
        sequence_number=None,  # would raise in _resolve_inheritance if reached
        df_content=0,
        df_file_path="s3://b/data/deleted.orc",
        df_file_format="ORC",
        df_partition=Row(part_0="X"),
        df_record_count=5,
        df_equality_ids=[],
        _manifest_path=_MANIFEST_A,
    )
    meta = _ml_row()
    data_files, delete_files = _build_scan_items([row], meta, None)
    assert data_files == []
    assert delete_files == []


def test_manifest_path_normalization_allows_scheme_mismatch():
    """_manifest_path with s3:// is found in ml_meta with s3a:// key via normalizer.

    In production, _metadata.file_path (from Spark) may return s3a:// while the
    manifest list stored s3://.  Both sides must be normalized for the lookup to
    succeed (spec §9).
    """
    from pyspark_iceberg_reader.paths import DEFAULT_NORMALIZER

    s3a_path = "s3a://test-bucket/metadata/manifest_a.avro"
    s3_path = "s3://test-bucket/metadata/manifest_a.avro"

    # ml_meta is pre-normalized (as built by resolve_manifests with the normalizer)
    ml_meta = {s3a_path: Row(
        sequence_number=10,
        added_snapshot_id=99,
        partition_spec_id=0,
        content=0,
    )}
    rows = [_entry(status=1, seq=None, snap_id=None, content=0,
                   path="s3://b/data/f1.parquet", manifest_path=s3_path)]

    data_files, _ = _build_scan_items(rows, ml_meta, DEFAULT_NORMALIZER)
    assert len(data_files) == 1
    assert data_files[0].data_sequence_number == 10  # inherited from ml_meta entry
