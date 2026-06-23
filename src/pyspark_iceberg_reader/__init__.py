"""pyspark-iceberg-reader: read Iceberg v2 tables via PySpark without a catalog.

    from pyspark_iceberg_reader import read_iceberg_table
    df = read_iceberg_table("s3://bucket/table/metadata/00001.metadata.json")
"""
from __future__ import annotations

from pyspark_iceberg_reader.reader import read_iceberg_table

__all__ = ["read_iceberg_table"]
