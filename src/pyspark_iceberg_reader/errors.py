"""Typed exceptions raised by pyspark-iceberg-reader.

Every public function raises one of these rather than a bare Exception so
callers can distinguish failure modes without string-matching error messages.
"""
from __future__ import annotations


class IcebergReaderError(Exception):
    """Base class for all pyspark-iceberg-reader errors."""


class UnsupportedFormatVersionError(IcebergReaderError):
    """Raised when ``format-version`` is not 2.

    :param version: The format version found in metadata.json.
    """

    def __init__(self, version: int) -> None:
        super().__init__(f"Unsupported Iceberg format version {version}; only v2 is supported.")
        self.version = version


class MetadataParseError(IcebergReaderError):
    """Raised when metadata.json is missing required fields or is otherwise malformed."""


class SnapshotNotFoundError(IcebergReaderError):
    """Raised when an explicit ``snapshot_id`` does not exist in the table's snapshot list.

    :param snapshot_id: The ID that was requested but not found.
    """

    def __init__(self, snapshot_id: int) -> None:
        super().__init__(f"Snapshot {snapshot_id} not found in table metadata.")
        self.snapshot_id = snapshot_id


class UnsupportedFeatureError(IcebergReaderError):
    """Raised when a valid but unsupported Iceberg feature is encountered.

    Examples: ORC data files, unrecognised partition transforms, corrupt
    manifest inheritance. Prefer this over silently returning wrong rows.
    """
