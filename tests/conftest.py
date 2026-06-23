"""Shared pytest fixtures.

The ``spark`` fixture starts a local-mode SparkSession.  It is session-scoped
so Spark initialises once per test run.  Tests that do not request the ``spark``
fixture pay no startup cost.
"""
from __future__ import annotations

import glob
import os

import pytest
import pyspark
from pyspark.sql import SparkSession

# spark-avro is bundled in Databricks Runtime but NOT in the pip pyspark package —
# pyspark[avro] only installs Python deps (pyarrow).  For local-mode tests we add
# the JVM JAR via spark.jars.packages so Spark downloads it from Maven once and
# caches it in ~/.ivy2.  If the JAR is already present in the pyspark installation
# (e.g. from a pyspark-avro wheel), we skip the network download.
_AVRO_MAVEN_COORD = "org.apache.spark:spark-avro_2.13:4.0.0"
_avro_bundled = bool(
    glob.glob(os.path.join(os.path.dirname(pyspark.__file__), "jars", "spark-avro*.jar"))
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Local-mode SparkSession for integration tests. No cluster required.

    :returns: A ``SparkSession`` running on ``local[2]``.
    """
    builder = (
        SparkSession.builder
        .master("local[2]")
        .appName("pyspark-iceberg-reader-tests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.parquet.fieldId.read.enabled", "true")
        # Required on Java 18+ where the security manager was removed; harmless on 17.
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    )
    if not _avro_bundled:
        builder = builder.config("spark.jars.packages", _AVRO_MAVEN_COORD)
    return builder.getOrCreate()
