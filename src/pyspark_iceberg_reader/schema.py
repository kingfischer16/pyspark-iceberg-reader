"""Convert Iceberg v2 schemas to Spark ``StructType`` with field-ID metadata.

Field-ID metadata (``{"parquet.field.id": id}``) on every ``StructField`` is
what makes schema-evolution-safe reads possible: Spark matches parquet columns
by ID instead of name, so renames and reorders resolve correctly and columns
added after a file was written read as ``null``.  Type promotions (e.g.
``int`` → ``long``) are handled automatically by Spark when reading with a
field-ID schema.

Spec §4.8, §7.4.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DataType,
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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# One lookup table for all fixed primitive type mappings (Spec §4.8).
# Parametric types (decimal, fixed) are handled separately via regex.
_PRIMITIVE_TYPES: dict[str, DataType] = {
    "boolean": BooleanType(),
    "int": IntegerType(),
    "long": LongType(),
    "float": FloatType(),
    "double": DoubleType(),
    "date": DateType(),
    # Iceberg time = microseconds since midnight.  Spark has no time-of-day
    # type, so we store as LongType and document the semantics.
    "time": LongType(),
    # Iceberg timestamp (no tz) maps to Spark TimestampNTZType (Spark 3.4+).
    "timestamp": TimestampNTZType(),
    # Iceberg timestamptz (UTC-normalised) maps to Spark TimestampType.
    "timestamptz": TimestampType(),
    "string": StringType(),
    # Iceberg uuid has no native Spark equivalent; store as string.
    "uuid": StringType(),
    "binary": BinaryType(),
}

# Regex for parametric decimal type: decimal(precision, scale)
_DECIMAL_RE: re.Pattern[str] = re.compile(r"^decimal\((\d+),\s*(\d+)\)$")

# Regex for fixed-length binary type: fixed[length]
_FIXED_RE: re.Pattern[str] = re.compile(r"^fixed\[(\d+)\]$")

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def iceberg_type_to_spark(field_type: str | dict[str, Any]) -> DataType:
    """Convert an Iceberg field type to the equivalent Spark ``DataType``.

    Handles all Iceberg v2 primitive and complex types.  Complex types
    (struct, list, map) are processed recursively so that nested struct
    fields carry their Iceberg field IDs in ``StructField.metadata``.

    :param field_type: Either a primitive type string (e.g. ``"long"``,
        ``"decimal(10, 2)"``) or a complex-type dict (``{"type": "struct",
        ...}``, ``{"type": "list", ...}``, ``{"type": "map", ...}``).
        This is the raw value from :attr:`~pyspark_iceberg_reader.metadata.IcebergField.field_type`.
    :returns: The corresponding Spark ``DataType``.
    :raises UnsupportedFeatureError: For type strings or complex-type names
        not in the Iceberg v2 type system.

    Spec §4.8, §7.4.
    """
    if isinstance(field_type, str):
        return _parse_primitive(field_type)

    type_name = field_type.get("type")
    if type_name == "struct":
        return _parse_struct_dict(field_type)
    if type_name == "list":
        element_spark = iceberg_type_to_spark(field_type["element"])
        contains_null = not field_type.get("element-required", True)
        return ArrayType(element_spark, containsNull=contains_null)
    if type_name == "map":
        key_spark = iceberg_type_to_spark(field_type["key"])
        value_spark = iceberg_type_to_spark(field_type["value"])
        value_contains_null = not field_type.get("value-required", True)
        return MapType(key_spark, value_spark, valueContainsNull=value_contains_null)

    raise UnsupportedFeatureError(
        f"Unsupported Iceberg complex type: {type_name!r}. "
        "Only struct, list, and map are supported in v2."
    )


def iceberg_schema_to_spark(schema: IcebergSchema) -> StructType:
    """Convert an :class:`~pyspark_iceberg_reader.metadata.IcebergSchema` to a
    Spark ``StructType`` with Iceberg field IDs embedded in each field's metadata.

    The returned schema is passed to ``spark.read.schema(...)`` together with
    ``spark.sql.parquet.fieldId.read.enabled = true`` so that Spark matches
    parquet columns by field ID rather than name (Spec §7.4, §4.8).

    :param schema: The Iceberg schema to convert.  Typically the table's
        *current* schema from
        :attr:`~pyspark_iceberg_reader.metadata.TableMetadata.schemas`.
    :returns: A ``StructType`` where every ``StructField`` carries
        ``metadata={"parquet.field.id": <field_id>}``.

    Spec §4.8, §7.4.
    """
    return StructType(
        [
            _build_struct_field(f.name, f.field_type, f.required, f.field_id)
            for f in schema.fields
        ]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_primitive(type_str: str) -> DataType:
    """Parse a primitive Iceberg type string to a Spark DataType."""
    if type_str in _PRIMITIVE_TYPES:
        return _PRIMITIVE_TYPES[type_str]
    m = _DECIMAL_RE.match(type_str)
    if m:
        return DecimalType(int(m.group(1)), int(m.group(2)))
    m = _FIXED_RE.match(type_str)
    if m:
        # fixed[L] → BinaryType; Spark does not have a fixed-length binary type.
        return BinaryType()
    raise UnsupportedFeatureError(
        f"Unsupported Iceberg primitive type: {type_str!r}."
    )


def _build_struct_field(
    name: str,
    field_type: str | dict[str, Any],
    required: bool,
    field_id: int,
) -> StructField:
    """Build a ``StructField`` with Iceberg field-ID metadata."""
    return StructField(
        name,
        iceberg_type_to_spark(field_type),
        nullable=not required,
        metadata={"parquet.field.id": field_id},
    )


def _parse_struct_dict(struct_dict: dict[str, Any]) -> StructType:
    """Parse a ``{"type": "struct", "fields": [...]}`` dict into a StructType."""
    try:
        return StructType(
            [
                _build_struct_field(f["name"], f["type"], f.get("required", False), f["id"])
                for f in struct_dict["fields"]
            ]
        )
    except KeyError as exc:
        raise MetadataParseError(
            f"Malformed nested struct field — missing required key {exc} "
            f"in type definition: {struct_dict!r}"
        ) from exc


__all__ = [
    "iceberg_type_to_spark",
    "iceberg_schema_to_spark",
]
