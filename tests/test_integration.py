"""End-to-end integration tests for pyspark-iceberg-reader on DBR 17.3.

These tests verify the full public API — read_iceberg_table() — against
real Iceberg tables with Avro manifests, run on Databricks serverless compute.

All tests require a live DBR 17.3 session and are excluded from local CI.

Run on cluster:
    pytest tests/test_integration.py -m cluster -v --tb=short

Spec §12 (testing strategy), §3 (runtime requirements).
"""
from __future__ import annotations

import pytest

pyiceberg = pytest.importorskip("pyiceberg", reason="pyiceberg required for integration fixtures")

import pandas.testing as tm
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.expressions import EqualTo
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, LongType, NestedField, StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark_iceberg_reader import read_iceberg_table
from pyspark_iceberg_reader.errors import SnapshotNotFoundError

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_SCHEMA_V1 = Schema(
    NestedField(1, "id", IntegerType(), required=False),
    NestedField(2, "name", StringType(), required=False),
)

_SCHEMA_V2 = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "name", StringType(), required=False),
    NestedField(3, "category", StringType(), required=False),
)

_SCHEMA_SIMPLE = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "name", StringType(), required=False),
)


# ---------------------------------------------------------------------------
# Fixture: generate Iceberg tables once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integration_tables(
    tmp_path_factory: pytest.TempPathFactory, spark: SparkSession
) -> dict[str, str]:
    """Generate deterministic Iceberg tables; return {name: metadata_location}.

    Uses PyIceberg's SqlCatalog (SQLite-backed, local warehouse) so tables
    are created with real Avro manifest lists and manifest files — the same
    file format that the native planner reads on DBR.
    """
    tmp = tmp_path_factory.mktemp("iceberg_integration")
    catalog = SqlCatalog(
        "integration",
        **{
            "uri": f"sqlite:///{tmp}/catalog.db",
            "warehouse": tmp.as_uri(),
        },
    )
    catalog.create_namespace("ns")
    tables: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. simple — 5 rows, single append, no deletes
    # ------------------------------------------------------------------
    t = catalog.create_table("ns.simple", schema=_SCHEMA_SIMPLE)
    t.append(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
                "name": pa.array(["alice", "bob", "carol", "dave", "eve"]),
            }
        )
    )
    tables["simple"] = t.metadata_location

    # ------------------------------------------------------------------
    # 2. eq_deletes — 5 rows + equality delete removing ids 2 and 4
    # ------------------------------------------------------------------
    t2 = catalog.create_table(
        "ns.eq_deletes",
        schema=_SCHEMA_SIMPLE,
        properties={"write.delete.mode": "merge-on-read"},
    )
    t2.append(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
                "name": pa.array(["alice", "bob", "carol", "dave", "eve"]),
            }
        )
    )
    t2.delete(EqualTo("id", 2))
    t2.delete(EqualTo("id", 4))
    tables["eq_deletes"] = t2.metadata_location

    # ------------------------------------------------------------------
    # 3. schema_evolved — written with v1 schema (int32 id, no category),
    #    table schema then promoted to v2 (int64 id, +category).
    #    Old rows read back with category=None; new rows have category set.
    # ------------------------------------------------------------------
    t3 = catalog.create_table("ns.schema_evolved", schema=_SCHEMA_V1)
    t3.append(
        pa.table(
            {
                "id": pa.array([1, 2], pa.int32()),
                "name": pa.array(["alice", "bob"]),
            }
        )
    )
    with t3.update_schema() as upd:
        upd.update_column("id", LongType())
        upd.add_column("category", StringType())
    t3.append(
        pa.table(
            {
                "id": pa.array([3, 4], pa.int64()),
                "name": pa.array(["carol", "dave"]),
                "category": pa.array(["x", "y"]),
            }
        )
    )
    tables["schema_evolved"] = t3.metadata_location

    # ------------------------------------------------------------------
    # 4. multi_snapshot — 3 sequential appends; tests explicit snapshot_id
    # ------------------------------------------------------------------
    t4 = catalog.create_table("ns.multi_snap", schema=_SCHEMA_SIMPLE)
    t4.append(
        pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "name": pa.array(["alice", "bob"]),
            }
        )
    )
    snap1_id = t4.current_snapshot().snapshot_id  # type: ignore[union-attr]
    t4.append(
        pa.table(
            {
                "id": pa.array([3, 4], pa.int64()),
                "name": pa.array(["carol", "dave"]),
            }
        )
    )
    t4.append(
        pa.table(
            {
                "id": pa.array([5, 6], pa.int64()),
                "name": pa.array(["eve", "frank"]),
            }
        )
    )
    tables["multi_snapshot"] = t4.metadata_location
    tables["multi_snapshot_snap1_id"] = snap1_id  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 5. empty — table with no data
    # ------------------------------------------------------------------
    t5 = catalog.create_table("ns.empty", schema=_SCHEMA_SIMPLE)
    _ = t5  # no appends
    tables["empty"] = t5.metadata_location

    return tables


# ---------------------------------------------------------------------------
# Runtime capability tests (extend test_runtime_capabilities.py stubs)
# ---------------------------------------------------------------------------


@pytest.mark.cluster
def test_avro_datasource_available(spark: SparkSession) -> None:
    """spark.read.format('avro') is registered on this DBR build."""
    try:
        spark.read.format("avro")
    except Exception as exc:
        pytest.fail(f"spark.read.format('avro') raised: {exc}")


@pytest.mark.cluster
def test_parquet_field_id_read_enabled(spark: SparkSession) -> None:
    """spark.sql.parquet.fieldId.read.enabled can be set and takes effect."""
    original = spark.conf.get("spark.sql.parquet.fieldId.read.enabled", "false")
    try:
        spark.conf.set("spark.sql.parquet.fieldId.read.enabled", "true")
        assert spark.conf.get("spark.sql.parquet.fieldId.read.enabled") == "true"
    finally:
        spark.conf.set("spark.sql.parquet.fieldId.read.enabled", original)


@pytest.mark.cluster
def test_metadata_row_index_sequential(spark: SparkSession, tmp_path) -> None:
    """_metadata.row_index is [0, 1, 2] for a single 3-row Parquet file.

    Sequential ordering is a hard requirement for position-delete correctness
    (spec §4.7).  This test goes beyond test_runtime_capabilities.py by writing
    the file via PyArrow (field IDs intact) and asserting the exact index values.
    """
    import pyarrow.parquet as pq

    path = tmp_path / "row_idx_check.parquet"
    pq.write_table(pa.table({"x": pa.array([10, 20, 30])}), str(path))

    df = spark.read.parquet(path.as_uri())
    indices = sorted(r._rid for r in df.withColumn("_rid", col("_metadata.row_index")).select("_rid").collect())
    assert indices == [0, 1, 2], f"Expected [0,1,2], got {indices}"


# ---------------------------------------------------------------------------
# Public API tests — read_iceberg_table()
# ---------------------------------------------------------------------------


@pytest.mark.cluster
def test_simple_read(integration_tables: dict, spark: SparkSession) -> None:
    """Simple append-only table returns all 5 rows with correct schema."""
    df = read_iceberg_table(integration_tables["simple"], spark=spark)
    rows = df.sort("id").collect()
    assert len(rows) == 5
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["name"] for r in rows] == ["alice", "bob", "carol", "dave", "eve"]
    assert "id" in df.columns
    assert "name" in df.columns


@pytest.mark.cluster
def test_equality_deletes(integration_tables: dict, spark: SparkSession) -> None:
    """Equality-deleted rows (ids 2, 4) are absent from the result."""
    df = read_iceberg_table(integration_tables["eq_deletes"], spark=spark)
    rows = df.sort("id").collect()
    ids = [r["id"] for r in rows]
    assert len(ids) == 3, f"Expected 3 rows, got {len(ids)}: {ids}"
    assert ids == [1, 3, 5]
    assert 2 not in ids
    assert 4 not in ids


@pytest.mark.cluster
def test_schema_evolution(integration_tables: dict, spark: SparkSession) -> None:
    """Rows written before schema evolution have category=None; newer rows have it set."""
    df = read_iceberg_table(integration_tables["schema_evolved"], spark=spark)
    rows = {r["id"]: r for r in df.collect()}
    assert set(rows.keys()) == {1, 2, 3, 4}
    assert rows[1]["category"] is None, "Pre-evolution row should have category=None"
    assert rows[2]["category"] is None, "Pre-evolution row should have category=None"
    assert rows[3]["category"] == "x"
    assert rows[4]["category"] == "y"
    # id should be promoted to int64
    assert df.schema["id"].dataType.simpleString() == "bigint"


@pytest.mark.cluster
def test_default_snapshot_is_current(integration_tables: dict, spark: SparkSession) -> None:
    """Default read of multi-snapshot table returns all 6 rows (latest snapshot)."""
    df = read_iceberg_table(integration_tables["multi_snapshot"], spark=spark)
    assert df.count() == 6


@pytest.mark.cluster
def test_explicit_snapshot_id(integration_tables: dict, spark: SparkSession) -> None:
    """Passing snapshot_id of the first snapshot returns only 2 rows."""
    snap1_id = integration_tables["multi_snapshot_snap1_id"]
    df = read_iceberg_table(
        integration_tables["multi_snapshot"],
        snapshot_id=snap1_id,
        spark=spark,
    )
    rows = df.sort("id").collect()
    assert len(rows) == 2, f"Expected 2 rows for snapshot 1, got {len(rows)}"
    assert [r["id"] for r in rows] == [1, 2]


@pytest.mark.cluster
def test_empty_table_returns_empty_df(integration_tables: dict, spark: SparkSession) -> None:
    """Table with no appends returns an empty DataFrame with the table schema."""
    df = read_iceberg_table(integration_tables["empty"], spark=spark)
    assert df.count() == 0
    assert "id" in df.columns
    assert "name" in df.columns


@pytest.mark.cluster
def test_bad_snapshot_id_raises(integration_tables: dict, spark: SparkSession) -> None:
    """A nonexistent snapshot_id raises SnapshotNotFoundError (spec §10)."""
    with pytest.raises(SnapshotNotFoundError):
        read_iceberg_table(
            integration_tables["simple"],
            snapshot_id=9_999_999_999,
            spark=spark,
        )


# ---------------------------------------------------------------------------
# Oracle comparison helpers
# ---------------------------------------------------------------------------


def _oracle_pdf(metadata_location: str, sort_by: str, snapshot_id: int | None = None):
    """Read via PyIceberg StaticTable; return sorted pandas DataFrame."""
    from pyiceberg.table import StaticTable
    table = StaticTable.from_metadata(metadata_location)
    scan = table.scan(snapshot_id=snapshot_id) if snapshot_id is not None else table.scan()
    return scan.to_arrow().to_pandas().sort_values(sort_by).reset_index(drop=True)


def _our_pdf(df, sort_by: str):
    """Convert our Spark DataFrame; return sorted pandas DataFrame."""
    return df.toPandas().sort_values(sort_by).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Oracle comparison tests — row-level parity against PyIceberg's read path
# ---------------------------------------------------------------------------


@pytest.mark.cluster
def test_oracle_simple_read(integration_tables: dict, spark: SparkSession) -> None:
    """Our simple read matches PyIceberg row-for-row."""
    loc = integration_tables["simple"]
    our = _our_pdf(read_iceberg_table(loc, spark=spark), sort_by="id")
    oracle = _oracle_pdf(loc, sort_by="id")
    tm.assert_frame_equal(our, oracle, check_dtype=False, check_like=True)


@pytest.mark.cluster
def test_oracle_equality_deletes(integration_tables: dict, spark: SparkSession) -> None:
    """Equality-delete application matches PyIceberg row-for-row."""
    loc = integration_tables["eq_deletes"]
    our = _our_pdf(read_iceberg_table(loc, spark=spark), sort_by="id")
    oracle = _oracle_pdf(loc, sort_by="id")
    tm.assert_frame_equal(our, oracle, check_dtype=False, check_like=True)


@pytest.mark.cluster
def test_oracle_schema_evolution(integration_tables: dict, spark: SparkSession) -> None:
    """Type promotion and null-fill for added column match PyIceberg row-for-row."""
    loc = integration_tables["schema_evolved"]
    our = _our_pdf(read_iceberg_table(loc, spark=spark), sort_by="id")
    oracle = _oracle_pdf(loc, sort_by="id")
    tm.assert_frame_equal(our, oracle, check_dtype=False, check_like=True)


@pytest.mark.cluster
def test_oracle_explicit_snapshot(integration_tables: dict, spark: SparkSession) -> None:
    """Time-travel to snapshot 1 matches PyIceberg row-for-row."""
    loc = integration_tables["multi_snapshot"]
    snap1_id = integration_tables["multi_snapshot_snap1_id"]
    our = _our_pdf(
        read_iceberg_table(loc, snapshot_id=snap1_id, spark=spark), sort_by="id"
    )
    oracle = _oracle_pdf(loc, sort_by="id", snapshot_id=snap1_id)
    tm.assert_frame_equal(our, oracle, check_dtype=False, check_like=True)
