"""Generate committed Parquet test fixtures for pyspark-iceberg-reader.

This is a one-time development utility — it is NOT run by CI.
Run locally once and commit the output under tests/fixtures/.

    py -3.14 -m pip install pyarrow
    py -3.14 scripts/generate_fixtures.py

Uses pyarrow only (pure Python, no Java/Spark needed).

All Parquet files are written with Iceberg field IDs embedded so that
``spark.sql.parquet.fieldId.read.enabled = true`` works correctly in
execution.py.  Field IDs are stored as ``PARQUET:field_id`` in pyarrow
field metadata (Parquet spec §4.8).

Output layout (all under tests/fixtures/):

    copy_on_write/data/part-0.parquet
        schema: id (int64, field_id=1), name (string, field_id=2)
        5 rows, no deletes — baseline for execute_scan_plan() smoke test

    equality_deletes/data/part-0.parquet
        schema: id (int64, fid=1), name (string, fid=2), category (string, fid=3)
        5 rows — id 2 and 4 will be equality-deleted

    equality_deletes/deletes/part-0.parquet
        schema: id (int64, fid=1)
        2 rows: {id: 2}, {id: 4}

    schema_evolved/v1_data/part-0.parquet
        schema: id (int32, fid=1), name (string, fid=2)  ← int32 promoted to int64

    schema_evolved/v2_data/part-0.parquet
        schema: id (int64, fid=1), name (string, fid=2), category (string, fid=3)
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent.parent
FIXTURES = _ROOT / "tests" / "fixtures"


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is required for fixture generation.")
        print("  py -3.14 -m pip install pyarrow")
        sys.exit(1)


def _field(name: str, pa_type, field_id: int):
    """Build a pyarrow field with an Iceberg-compatible field ID."""
    import pyarrow as pa
    return pa.field(name, pa_type).with_metadata({"PARQUET:field_id": str(field_id)})


def write_copy_on_write() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = FIXTURES / "copy_on_write" / "data"
    dest.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        _field("id", pa.int64(), 1),
        _field("name", pa.string(), 2),
    ])
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "name": pa.array(["alice", "bob", "carol", "dave", "eve"]),
        },
        schema=schema,
    )
    pq.write_table(table, dest / "part-0.parquet")
    print("OK copy_on_write/data/part-0.parquet (5 rows, field IDs: id=1, name=2)")


def write_equality_deletes() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dest = FIXTURES / "equality_deletes" / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    data_schema = pa.schema([
        _field("id", pa.int64(), 1),
        _field("name", pa.string(), 2),
        _field("category", pa.string(), 3),
    ])
    pq.write_table(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
                "name": pa.array(["alice", "bob", "carol", "dave", "eve"]),
                "category": pa.array(["A", "A", "B", "B", "A"]),
            },
            schema=data_schema,
        ),
        data_dest / "part-0.parquet",
    )
    print("OK equality_deletes/data/part-0.parquet (5 rows, field IDs: id=1, name=2, category=3)")

    del_dest = FIXTURES / "equality_deletes" / "deletes"
    del_dest.mkdir(parents=True, exist_ok=True)
    del_schema = pa.schema([_field("id", pa.int64(), 1)])
    pq.write_table(
        pa.table({"id": pa.array([2, 4], type=pa.int64())}, schema=del_schema),
        del_dest / "part-0.parquet",
    )
    print("OK equality_deletes/deletes/part-0.parquet (2 delete rows: id 2, 4)")


def write_schema_evolved() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    v1_dest = FIXTURES / "schema_evolved" / "v1_data"
    v1_dest.mkdir(parents=True, exist_ok=True)
    # id is int32 here — current schema uses int64; field ID=1 so Spark matches by ID
    v1_schema = pa.schema([
        _field("id", pa.int32(), 1),
        _field("name", pa.string(), 2),
    ])
    pq.write_table(
        pa.table(
            {
                "id": pa.array([1, 2], type=pa.int32()),
                "name": pa.array(["alice", "bob"]),
            },
            schema=v1_schema,
        ),
        v1_dest / "part-0.parquet",
    )
    print("OK schema_evolved/v1_data/part-0.parquet (2 rows, id=int32 fid=1, name fid=2)")

    v2_dest = FIXTURES / "schema_evolved" / "v2_data"
    v2_dest.mkdir(parents=True, exist_ok=True)
    v2_schema = pa.schema([
        _field("id", pa.int64(), 1),
        _field("name", pa.string(), 2),
        _field("category", pa.string(), 3),
    ])
    pq.write_table(
        pa.table(
            {
                "id": pa.array([3, 4], type=pa.int64()),
                "name": pa.array(["carol", "dave"]),
                "category": pa.array(["X", "Y"]),
            },
            schema=v2_schema,
        ),
        v2_dest / "part-0.parquet",
    )
    print("OK schema_evolved/v2_data/part-0.parquet (2 rows, id=int64 fid=1, category fid=3)")


if __name__ == "__main__":
    _require_pyarrow()
    write_copy_on_write()
    write_equality_deletes()
    write_schema_evolved()
    print(f"\nAll fixtures written to {FIXTURES}")
