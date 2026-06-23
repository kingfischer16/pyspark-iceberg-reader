"""Path scheme normalisation and relative-path resolution.

Iceberg stores absolute paths in manifests. Schemes may differ between what the writer
recorded and what the running cluster mounts (e.g. ``s3`` vs ``s3a``). This module
provides an optional normalisation hook applied symmetrically to both sides of delete
joins so path comparisons succeed regardless of writer convention.

Spec §9.
"""
from __future__ import annotations

import logging
import posixpath
from collections.abc import Callable

from pyspark_iceberg_reader.errors import UnsupportedFeatureError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scheme remaps applied by DEFAULT_NORMALIZER.  One lookup table — add new
# mappings here only.
_SCHEME_MAP: dict[str, str] = {
    "s3": "s3a",
    "wasbs": "abfss",
}

# Every scheme the codebase has documented.  Used only by validate_scheme.
_KNOWN_SCHEMES: frozenset[str] = frozenset(
    {"s3", "s3a", "dbfs", "abfss", "wasbs", "gs", "file"}
)

# Type alias for the normalisation hook.
PathNormalizer = Callable[[str], str]


# ---------------------------------------------------------------------------
# Built-in normaliser
# ---------------------------------------------------------------------------


def DEFAULT_NORMALIZER(path: str) -> str:  # noqa: N802  (uppercase intentional — treated as a named constant)
    """Normalise common Iceberg path scheme variants so paths compare equal.

    Rules (Spec §9):

    - ``s3://`` → ``s3a://`` (Spark/Hadoop prefers ``s3a``).
    - ``wasbs://`` → ``abfss://`` (scheme only; host is left unchanged — Azure
      ``.blob.`` vs ``.dfs.`` host remapping requires a custom normaliser).
    - ``dbfs://`` → ``dbfs:/`` (strip the spurious second slash that some
      writers emit).
    - All other schemes, including unknown ones, are returned unchanged.

    This function is defined at module scope (not as a lambda or closure) so
    that it is picklable and can be registered as a Spark UDF by
    ``execution.py``.
    """
    sep = "://"
    idx = path.find(sep)
    if idx == -1:
        return path

    scheme = path[:idx]
    rest = path[idx + len(sep):]  # everything after "://"

    canonical_scheme = _SCHEME_MAP.get(scheme, scheme)

    # dbfs://rest → dbfs:/rest  (collapse the spurious double-slash authority).
    # rest is everything after "://", e.g. "mnt/table/f.parquet".
    # Canonical dbfs paths use a single slash: dbfs:/mnt/…
    # NOTE: dbfs is handled here, not in _SCHEME_MAP, because _SCHEME_MAP only
    # remaps scheme names (e.g. s3→s3a).  For dbfs the scheme stays "dbfs" but
    # the URL structure changes (double-slash authority → single-slash).
    # Add new scheme remaps to _SCHEME_MAP; add structural transforms here.
    if canonical_scheme == "dbfs":
        return f"dbfs:/{rest}"

    return f"{canonical_scheme}://{rest}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalize_path(path: str, normalizer: PathNormalizer | None = None) -> str:
    """Apply *normalizer* to *path*, or return *path* unchanged if normalizer is None.

    This is the single named seam for §9 normalisation in the codebase.  All
    code that needs to normalise a path imports and calls this function so that
    future changes (logging, metrics) are in one place.

    ``file:///path`` (RFC 3986 empty authority) is always collapsed to
    ``file:/path`` before the custom normaliser runs.  Both forms are valid
    representations of the same local file; Spark's ``_metadata.file_path``
    uses the single-slash form, while some writers (e.g. PyIceberg) use the
    triple-slash form, so normalising unconditionally ensures comparisons succeed.

    :param path: The path string to normalise.
    :param normalizer: A callable ``str → str``.  Pass ``None`` (the default)
        to use paths verbatim, ``DEFAULT_NORMALIZER`` for built-in scheme
        remapping, or a custom callable for mount-level remapping.
    :returns: The normalised path, or *path* unchanged when *normalizer* is
        ``None``.

    Spec §9.
    """
    if path.startswith("file:///"):
        path = "file:/" + path[8:]
    if normalizer is None:
        return path
    return normalizer(path)


def _has_scheme(path: str) -> bool:
    """Return True if *path* has a URI scheme (colon before the first slash).

    Handles both double-slash (``s3://``) and single-slash (``dbfs:/``) forms.
    """
    colon = path.find(":")
    if colon == -1:
        return False
    slash = path.find("/")
    return slash == -1 or colon < slash


def resolve_path(path: str, base_location: str) -> str:
    """Resolve *path* against *base_location* if it is relative.

    Absolute paths — those with any URI scheme (e.g. ``s3://``, ``dbfs:/``,
    ``abfss://``) or starting with ``/`` — are returned unchanged.  Relative
    paths (rare but spec-legal) are joined with *base_location* using POSIX
    semantics.

    .. note::
        ``posixpath.normpath`` collapses ``://`` into ``:/``, corrupting cloud
        URIs.  This function applies ``normpath`` only to the path component
        after the authority so the scheme is preserved.

    :param path: The path to resolve.  Usually from a manifest entry.
    :param base_location: The table's ``location`` field from metadata.json.
        Always an absolute path.
    :returns: The resolved absolute path.

    Spec §9.
    """
    if path.startswith("/") or _has_scheme(path):
        return path

    if "://" in base_location:
        # Split base_location into "scheme://authority" and the path component.
        scheme_auth, _, base_rest = base_location.partition("://")
        authority, _, path_rest = base_rest.partition("/")
        # Only normpath the path component — never the full URI string.
        joined = posixpath.normpath("/" + path_rest + "/" + path)
        return f"{scheme_auth}://{authority}{joined}"

    # base_location is an absolute POSIX path (no scheme).
    return posixpath.normpath(posixpath.join(base_location, path))


def validate_scheme(path: str) -> None:
    """Raise :class:`~pyspark_iceberg_reader.errors.UnsupportedFeatureError` if
    the scheme in *path* is not one of the documented Iceberg filesystem schemes.

    This is an **opt-in** strict check.  The normaliser itself never raises;
    callers that want to fail fast on unknown schemes call this separately.
    Paths with no scheme (relative paths, bare POSIX paths) pass silently.

    :param path: The path to validate.
    :raises UnsupportedFeatureError: When the scheme is not in
        :data:`_KNOWN_SCHEMES`.

    Spec §9.
    """
    if not _has_scheme(path):
        return
    scheme = path[:path.find(":")]
    if scheme not in _KNOWN_SCHEMES:
        raise UnsupportedFeatureError(
            f"Unknown filesystem scheme '{scheme}' in path: {path!r}. "
            f"Supported schemes: {sorted(_KNOWN_SCHEMES)}."
        )


__all__ = [
    "PathNormalizer",
    "DEFAULT_NORMALIZER",
    "normalize_path",
    "resolve_path",
    "validate_scheme",
]
