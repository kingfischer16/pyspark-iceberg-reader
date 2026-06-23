"""Tests for paths.py — path scheme normalisation and relative-path resolution.

All tests are pure Python; no Spark session required.

Spec §9.
"""
from __future__ import annotations

import pytest

from pyspark_iceberg_reader.errors import UnsupportedFeatureError
from pyspark_iceberg_reader.paths import (
    DEFAULT_NORMALIZER,
    normalize_path,
    resolve_path,
    validate_scheme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefix_normalizer(path: str) -> str:
    """Named test double that prepends 'TEST:' — proves normalize_path calls it."""
    return f"TEST:{path}"


# ---------------------------------------------------------------------------
# DEFAULT_NORMALIZER — scheme remapping
# ---------------------------------------------------------------------------


def test_default_normalizer_s3_to_s3a() -> None:
    assert DEFAULT_NORMALIZER("s3://bucket/key") == "s3a://bucket/key"


def test_default_normalizer_s3a_unchanged() -> None:
    assert DEFAULT_NORMALIZER("s3a://bucket/key") == "s3a://bucket/key"


def test_default_normalizer_wasbs_to_abfss_scheme_only() -> None:
    # Scheme remapped; host left unchanged — Azure .blob. vs .dfs. host
    # remapping requires a custom normaliser (Spec §9).
    result = DEFAULT_NORMALIZER("wasbs://container@account.blob.core.windows.net/path")
    assert result == "abfss://container@account.blob.core.windows.net/path"


def test_default_normalizer_abfss_unchanged() -> None:
    path = "abfss://container@account.dfs.core.windows.net/path"
    assert DEFAULT_NORMALIZER(path) == path


def test_default_normalizer_dbfs_double_slash_collapsed() -> None:
    assert DEFAULT_NORMALIZER("dbfs://mnt/table/file.parquet") == "dbfs:/mnt/table/file.parquet"


def test_default_normalizer_dbfs_single_slash_unchanged() -> None:
    assert DEFAULT_NORMALIZER("dbfs:/mnt/table/file.parquet") == "dbfs:/mnt/table/file.parquet"


def test_default_normalizer_unknown_scheme_passthrough() -> None:
    assert DEFAULT_NORMALIZER("gs://bucket/key") == "gs://bucket/key"


def test_default_normalizer_no_scheme_passthrough() -> None:
    assert DEFAULT_NORMALIZER("relative/path/file.parquet") == "relative/path/file.parquet"


def test_default_normalizer_s3_preserves_full_path() -> None:
    result = DEFAULT_NORMALIZER("s3://my-bucket/a/b/c.parquet")
    assert result == "s3a://my-bucket/a/b/c.parquet"


def test_default_normalizer_scheme_case_sensitive() -> None:
    # Scheme matching is case-sensitive; cloud paths use lowercase in practice.
    # Document the behaviour explicitly to avoid a future regression.
    assert DEFAULT_NORMALIZER("S3://bucket/key") == "S3://bucket/key"


def test_default_normalizer_empty_string() -> None:
    assert DEFAULT_NORMALIZER("") == ""


# ---------------------------------------------------------------------------
# normalize_path — dispatch
# ---------------------------------------------------------------------------


def test_normalize_path_none_normalizer_returns_unchanged() -> None:
    assert normalize_path("s3://bucket/key", None) == "s3://bucket/key"


def test_normalize_path_default_normalizer_remaps_s3() -> None:
    assert normalize_path("s3://bucket/key", DEFAULT_NORMALIZER) == "s3a://bucket/key"


def test_normalize_path_custom_normalizer_called() -> None:
    result = normalize_path("s3://bucket/key", _prefix_normalizer)
    assert result == "TEST:s3://bucket/key"


def test_normalize_path_empty_string_no_crash() -> None:
    assert normalize_path("", None) == ""


# ---------------------------------------------------------------------------
# resolve_path — absolute paths returned unchanged
# ---------------------------------------------------------------------------


def test_resolve_path_s3_absolute_unchanged() -> None:
    path = "s3://bucket/table/data/file.parquet"
    assert resolve_path(path, "s3://other-bucket/base") == path


def test_resolve_path_s3a_absolute_unchanged() -> None:
    path = "s3a://bucket/table/data/file.parquet"
    assert resolve_path(path, "s3a://other-bucket/base") == path


def test_resolve_path_abfss_absolute_unchanged() -> None:
    path = "abfss://container@host.dfs.core.windows.net/table/file.parquet"
    assert resolve_path(path, "abfss://container@host.dfs.core.windows.net/other") == path


def test_resolve_path_dbfs_absolute_unchanged() -> None:
    path = "dbfs:/mnt/table/data/file.parquet"
    assert resolve_path(path, "dbfs:/mnt/other") == path


def test_resolve_path_posix_absolute_unchanged() -> None:
    path = "/mnt/table/data/file.parquet"
    assert resolve_path(path, "s3://bucket/base") == path


# ---------------------------------------------------------------------------
# resolve_path — relative paths resolved against base
# ---------------------------------------------------------------------------


def test_resolve_path_relative_against_s3_base() -> None:
    result = resolve_path("data/00001.parquet", "s3://bucket/table")
    assert result == "s3://bucket/table/data/00001.parquet"


def test_resolve_path_relative_single_component() -> None:
    result = resolve_path("file.parquet", "s3a://bucket/prefix")
    assert result == "s3a://bucket/prefix/file.parquet"


def test_resolve_path_relative_dotslash_stripped() -> None:
    result = resolve_path("./data/file.parquet", "s3://bucket/table")
    assert result == "s3://bucket/table/data/file.parquet"


def test_resolve_path_relative_parent_traversal() -> None:
    result = resolve_path("../sibling/file.parquet", "s3://bucket/table/data")
    assert result == "s3://bucket/table/sibling/file.parquet"


def test_resolve_path_base_with_trailing_slash_no_double_slash() -> None:
    result = resolve_path("data/f.parquet", "s3://bucket/table/")
    assert "//" not in result.split("://", 1)[1]
    assert result == "s3://bucket/table/data/f.parquet"


def test_resolve_path_scheme_preserved_in_result() -> None:
    # Guard against posixpath.normpath collapsing :// into :/
    result = resolve_path("data/f.parquet", "s3://bucket/table")
    assert result.startswith("s3://")


# ---------------------------------------------------------------------------
# validate_scheme — known schemes pass; unknown raise
# ---------------------------------------------------------------------------


def test_validate_scheme_s3_passes() -> None:
    validate_scheme("s3://bucket/key")  # no exception


def test_validate_scheme_s3a_passes() -> None:
    validate_scheme("s3a://bucket/key")


def test_validate_scheme_dbfs_passes() -> None:
    validate_scheme("dbfs:/mnt/path")


def test_validate_scheme_abfss_passes() -> None:
    validate_scheme("abfss://container@host.dfs.core.windows.net/path")


def test_validate_scheme_wasbs_passes() -> None:
    validate_scheme("wasbs://container@host.blob.core.windows.net/path")


def test_validate_scheme_gs_passes() -> None:
    validate_scheme("gs://bucket/key")


def test_validate_scheme_file_passes() -> None:
    validate_scheme("file:///local/path")


def test_validate_scheme_unknown_raises() -> None:
    with pytest.raises(UnsupportedFeatureError):
        validate_scheme("hdfs://namenode/path")


def test_validate_scheme_error_message_contains_scheme() -> None:
    with pytest.raises(UnsupportedFeatureError, match="hdfs"):
        validate_scheme("hdfs://namenode/path")


def test_validate_scheme_relative_path_passes() -> None:
    validate_scheme("relative/path/file.parquet")  # no exception


def test_validate_scheme_empty_string_passes() -> None:
    validate_scheme("")  # no exception


def test_validate_scheme_single_slash_unknown_scheme_raises() -> None:
    # dbfs:/… form uses a single slash — validate_scheme must still catch
    # unknown single-slash schemes like foo:/bar (not just foo://bar).
    with pytest.raises(UnsupportedFeatureError, match="foo"):
        validate_scheme("foo:/bar/table")


def test_validate_scheme_dbfs_single_slash_passes() -> None:
    # dbfs:/mnt/… is the canonical form on Databricks; must not raise.
    validate_scheme("dbfs:/mnt/delta/table")


# ---------------------------------------------------------------------------
# Symmetric normalisation — the §7.6 delete-join scenario
# ---------------------------------------------------------------------------


def test_symmetric_s3_vs_s3a_equal_after_normalize() -> None:
    """Writer used s3://, cluster uses s3a:// — the canonical delete-join case."""
    data_path = normalize_path("s3://bucket/data/file.parquet", DEFAULT_NORMALIZER)
    delete_path = normalize_path("s3a://bucket/data/file.parquet", DEFAULT_NORMALIZER)
    assert data_path == delete_path


def test_symmetric_dbfs_double_vs_single_slash_equal() -> None:
    data_path = normalize_path("dbfs://mnt/table/file.parquet", DEFAULT_NORMALIZER)
    delete_path = normalize_path("dbfs:/mnt/table/file.parquet", DEFAULT_NORMALIZER)
    assert data_path == delete_path


def test_symmetric_identical_paths_remain_equal() -> None:
    path = "s3a://bucket/data/file.parquet"
    assert normalize_path(path, DEFAULT_NORMALIZER) == normalize_path(path, DEFAULT_NORMALIZER)


def test_symmetric_different_buckets_remain_unequal() -> None:
    path_a = normalize_path("s3://bucket-a/file.parquet", DEFAULT_NORMALIZER)
    path_b = normalize_path("s3://bucket-b/file.parquet", DEFAULT_NORMALIZER)
    assert path_a != path_b
