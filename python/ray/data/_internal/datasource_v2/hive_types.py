"""Hive result type normalization for the Ray Data Hive datasource."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence, Tuple

import pyarrow as pa

MAX_HIVE_COLUMNS = 8192

_TYPE_PARAMETER_RE = re.compile(r"^(CHAR|VARCHAR)\s*\(\s*\d+\s*\)$", re.I)
_DECIMAL_RE = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.I)


class HiveTypeCompatibilityError(ValueError):
    """Raised when an HS2 type has no unambiguous Arrow representation."""


def hive_type_to_arrow(type_code: Any, column_name: str) -> pa.DataType:
    if not isinstance(type_code, str):
        raise HiveTypeCompatibilityError(
            f"Hive column {column_name!r} has invalid type metadata {type_code!r}"
        )

    normalized = type_code.strip().upper()
    parameterized = _TYPE_PARAMETER_RE.fullmatch(normalized)
    if parameterized:
        normalized = parameterized.group(1)

    fixed_types = {
        "BOOLEAN": pa.bool_(),
        "TINYINT": pa.int8(),
        "SMALLINT": pa.int16(),
        "INT": pa.int32(),
        "INTEGER": pa.int32(),
        "BIGINT": pa.int64(),
        "FLOAT": pa.float32(),
        "DOUBLE": pa.float64(),
        "STRING": pa.string(),
        "CHAR": pa.string(),
        "VARCHAR": pa.string(),
        "BINARY": pa.binary(),
        "DATE": pa.date32(),
        "TIMESTAMP": pa.timestamp("ns"),
    }
    if normalized in fixed_types:
        return fixed_types[normalized]

    decimal_match = _DECIMAL_RE.fullmatch(normalized)
    if decimal_match:
        precision, scale = map(int, decimal_match.groups())
        if 1 <= precision <= 38 and 0 <= scale <= precision:
            return pa.decimal128(precision, scale)
        raise HiveTypeCompatibilityError(
            f"Hive column {column_name!r} has unsupported decimal type {type_code!r}"
        )

    raise HiveTypeCompatibilityError(
        f"Hive column {column_name!r} has unsupported or ambiguous type {type_code!r}"
    )


def schema_from_description(description: Sequence[Any]) -> pa.Schema:
    if not description:
        raise HiveTypeCompatibilityError(
            "Hive result metadata did not contain any column descriptions"
        )

    if len(description) > MAX_HIVE_COLUMNS:
        raise HiveTypeCompatibilityError(
            f"Hive result has more than the supported {MAX_HIVE_COLUMNS} columns"
        )

    fields = []
    names_seen = set()
    for index, column in enumerate(description):
        try:
            name = str(column[0])
            type_code = column[1]
        except (IndexError, KeyError, TypeError, AttributeError) as exc:
            raise HiveTypeCompatibilityError(
                f"Hive result metadata for column {index} is malformed"
            ) from exc

        if name in names_seen:
            raise HiveTypeCompatibilityError(
                f"Hive result contains duplicate column name {name!r}"
            )
        names_seen.add(name)
        if isinstance(type_code, str) and type_code.strip().upper() == "DECIMAL":
            precision = column[4] if len(column) > 4 else None
            scale = column[5] if len(column) > 5 else None
            if (
                isinstance(precision, int)
                and isinstance(scale, int)
                and 1 <= precision <= 38
                and 0 <= scale <= precision
            ):
                data_type = pa.decimal128(precision, scale)
            else:
                raise HiveTypeCompatibilityError(
                    f"Hive column {name!r} has decimal metadata without "
                    "precision and scale"
                )
        else:
            data_type = hive_type_to_arrow(type_code, name)
        fields.append(pa.field(name, data_type))
    return pa.schema(fields)


def schema_from_table_columns(columns: Iterable[Tuple[str, str]]) -> pa.Schema:
    fields = []
    names_seen = set()
    for name, type_code in columns:
        if len(fields) >= MAX_HIVE_COLUMNS:
            raise HiveTypeCompatibilityError(
                f"Hive relation has more than the supported {MAX_HIVE_COLUMNS} columns"
            )
        if name in names_seen:
            raise HiveTypeCompatibilityError(
                f"Hive relation contains duplicate column name {name!r}"
            )
        names_seen.add(name)
        fields.append(pa.field(name, hive_type_to_arrow(type_code, name)))
    if not fields:
        raise HiveTypeCompatibilityError("Hive relation metadata contains no columns")
    return pa.schema(fields)


def schemas_match(actual: pa.Schema, expected: pa.Schema) -> bool:
    return actual.names == expected.names and actual.types == expected.types
