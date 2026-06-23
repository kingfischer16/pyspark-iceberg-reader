"""Tests for schema.py — Iceberg → Spark type conversion.

Sections:
  - iceberg_type_to_spark — primitives  (pure Python, no Spark session needed)
  - iceberg_type_to_spark — complex types
  - iceberg_schema_to_spark             (Spark session used to verify real usage)

Spec §4.8, §7.4.
"""
from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

from pyspark_iceberg_reader.errors import MetadataParseError, UnsupportedFeatureError
from pyspark_iceberg_reader.metadata import IcebergField, IcebergSchema
from pyspark_iceberg_reader.schema import (
    iceberg_schema_to_spark,
    iceberg_type_to_spark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field(
    field_id: int,
    name: str,
    field_type: str | dict,
    required: bool = False,
) -> IcebergField:
    return IcebergField(field_id=field_id, name=name, field_type=field_type, required=required)


def _schema(*fields: IcebergField, schema_id: int = 0) -> IcebergSchema:
    return IcebergSchema(schema_id=schema_id, fields=tuple(fields))


# ---------------------------------------------------------------------------
# iceberg_type_to_spark — primitives
# ---------------------------------------------------------------------------


def test_type_boolean() -> None:
    assert iceberg_type_to_spark("boolean") == BooleanType()


def test_type_int() -> None:
    assert iceberg_type_to_spark("int") == IntegerType()


def test_type_long() -> None:
    assert iceberg_type_to_spark("long") == LongType()


def test_type_float() -> None:
    assert iceberg_type_to_spark("float") == FloatType()


def test_type_double() -> None:
    assert iceberg_type_to_spark("double") == DoubleType()


def test_type_date() -> None:
    assert iceberg_type_to_spark("date") == DateType()


def test_type_time_is_long() -> None:
    # Iceberg time = microseconds since midnight; no Spark time-of-day type.
    assert iceberg_type_to_spark("time") == LongType()


def test_type_timestamp_is_ntz() -> None:
    # Iceberg timestamp (no tz) → TimestampNTZType (Spark 3.4+).
    assert iceberg_type_to_spark("timestamp") == TimestampNTZType()


def test_type_timestamptz_is_tz() -> None:
    # Iceberg timestamptz (UTC-normalised) → TimestampType.
    assert iceberg_type_to_spark("timestamptz") == TimestampType()


def test_type_string() -> None:
    assert iceberg_type_to_spark("string") == StringType()


def test_type_uuid_is_string() -> None:
    # No native UUID type in Spark.
    assert iceberg_type_to_spark("uuid") == StringType()


def test_type_binary() -> None:
    assert iceberg_type_to_spark("binary") == BinaryType()


def test_type_decimal_with_precision_and_scale() -> None:
    assert iceberg_type_to_spark("decimal(10, 2)") == DecimalType(10, 2)


def test_type_decimal_no_space_after_comma() -> None:
    assert iceberg_type_to_spark("decimal(38,10)") == DecimalType(38, 10)


def test_type_fixed_is_binary() -> None:
    # fixed[L] has no fixed-length equivalent in Spark; store as BinaryType.
    assert iceberg_type_to_spark("fixed[16]") == BinaryType()


def test_type_unknown_primitive_raises() -> None:
    with pytest.raises(UnsupportedFeatureError, match="unknown_type"):
        iceberg_type_to_spark("unknown_type")


# ---------------------------------------------------------------------------
# iceberg_type_to_spark — complex types
# ---------------------------------------------------------------------------


def test_type_list_of_string() -> None:
    type_dict = {"type": "list", "element-id": 3, "element": "string", "element-required": True}
    result = iceberg_type_to_spark(type_dict)
    assert result == ArrayType(StringType(), containsNull=False)


def test_type_list_element_required_false_contains_null() -> None:
    type_dict = {"type": "list", "element-id": 3, "element": "int", "element-required": False}
    result = iceberg_type_to_spark(type_dict)
    assert result == ArrayType(IntegerType(), containsNull=True)


def test_type_list_of_struct_recursive_field_ids() -> None:
    type_dict = {
        "type": "list",
        "element-id": 5,
        "element": {
            "type": "struct",
            "fields": [
                {"id": 10, "name": "lat", "type": "float", "required": False},
                {"id": 11, "name": "lon", "type": "float", "required": False},
            ],
        },
        "element-required": False,
    }
    result = iceberg_type_to_spark(type_dict)
    assert isinstance(result, ArrayType)
    assert isinstance(result.elementType, StructType)
    assert result.elementType["lat"].metadata["parquet.field.id"] == 10
    assert result.elementType["lon"].metadata["parquet.field.id"] == 11


def test_type_map_string_to_int() -> None:
    type_dict = {
        "type": "map",
        "key-id": 4,
        "key": "string",
        "value-id": 5,
        "value": "int",
        "value-required": False,
    }
    result = iceberg_type_to_spark(type_dict)
    assert result == MapType(StringType(), IntegerType(), valueContainsNull=True)


def test_type_map_value_required_false_contains_null() -> None:
    type_dict = {
        "type": "map",
        "key-id": 4,
        "key": "string",
        "value-id": 5,
        "value": "long",
        "value-required": True,
    }
    result = iceberg_type_to_spark(type_dict)
    assert result == MapType(StringType(), LongType(), valueContainsNull=False)


def test_type_struct_recursive() -> None:
    type_dict = {
        "type": "struct",
        "fields": [
            {"id": 20, "name": "x", "type": "int", "required": True},
            {"id": 21, "name": "y", "type": "string", "required": False},
        ],
    }
    result = iceberg_type_to_spark(type_dict)
    assert isinstance(result, StructType)
    assert result["x"].metadata["parquet.field.id"] == 20
    assert result["x"].nullable is False
    assert result["y"].metadata["parquet.field.id"] == 21
    assert result["y"].nullable is True


def test_type_unknown_complex_raises() -> None:
    with pytest.raises(UnsupportedFeatureError, match="variant"):
        iceberg_type_to_spark({"type": "variant"})


def test_type_struct_missing_id_raises_metadata_parse_error() -> None:
    # A nested struct field missing the "id" key should raise MetadataParseError,
    # not a raw KeyError that leaks implementation details.
    type_dict = {
        "type": "struct",
        "fields": [
            {"name": "x", "type": "int"},  # missing "id"
        ],
    }
    with pytest.raises(MetadataParseError, match="id"):
        iceberg_type_to_spark(type_dict)


# ---------------------------------------------------------------------------
# iceberg_schema_to_spark  (Spark session verifies schema is usable in practice)
# ---------------------------------------------------------------------------


def test_schema_to_spark_field_id_in_metadata(spark: SparkSession) -> None:
    schema = _schema(
        _field(1, "id", "long", required=True),
        _field(2, "name", "string"),
    )
    st = iceberg_schema_to_spark(schema)
    assert st["id"].metadata["parquet.field.id"] == 1
    assert st["name"].metadata["parquet.field.id"] == 2


def test_schema_to_spark_required_field_is_not_nullable(spark: SparkSession) -> None:
    schema = _schema(_field(1, "pk", "int", required=True))
    st = iceberg_schema_to_spark(schema)
    assert st["pk"].nullable is False


def test_schema_to_spark_optional_field_is_nullable(spark: SparkSession) -> None:
    schema = _schema(_field(1, "opt", "string", required=False))
    st = iceberg_schema_to_spark(schema)
    assert st["opt"].nullable is True


def test_schema_to_spark_nested_struct_inner_fields_have_ids(spark: SparkSession) -> None:
    nested_type = {
        "type": "struct",
        "fields": [
            {"id": 10, "name": "lat", "type": "double", "required": False},
            {"id": 11, "name": "lon", "type": "double", "required": False},
        ],
    }
    schema = _schema(_field(1, "coords", nested_type))
    st = iceberg_schema_to_spark(schema)
    inner = st["coords"].dataType
    assert isinstance(inner, StructType)
    assert inner["lat"].metadata["parquet.field.id"] == 10
    assert inner["lon"].metadata["parquet.field.id"] == 11


def test_schema_to_spark_usable_in_create_dataframe(spark: SparkSession) -> None:
    schema = _schema(
        _field(1, "id", "long", required=True),
        _field(2, "value", "double"),
    )
    st = iceberg_schema_to_spark(schema)
    df = spark.createDataFrame([], st)
    assert df.schema == st


def test_schema_to_spark_all_primitive_types(spark: SparkSession) -> None:
    fields = [
        _field(1, "c_bool", "boolean"),
        _field(2, "c_int", "int"),
        _field(3, "c_long", "long"),
        _field(4, "c_float", "float"),
        _field(5, "c_double", "double"),
        _field(6, "c_date", "date"),
        _field(7, "c_str", "string"),
        _field(8, "c_binary", "binary"),
        _field(9, "c_decimal", "decimal(18, 4)"),
    ]
    schema = _schema(*fields)
    st = iceberg_schema_to_spark(schema)
    assert len(st.fields) == len(fields)
    for sf in st.fields:
        assert "parquet.field.id" in sf.metadata

