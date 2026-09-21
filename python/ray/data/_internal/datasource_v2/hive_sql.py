"""Conservative SQL generation for Hive table reads and filter pushdown."""

from __future__ import annotations

import datetime
import math
from decimal import Decimal
from typing import Optional, Tuple

import pyarrow as pa

from ray.data.expressions import (
    AliasExpr,
    BinaryExpr,
    ColumnExpr,
    Expr,
    LiteralExpr,
    Operation,
    UnaryExpr,
)

_MAX_IN_LIST_VALUES = 1000


def quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def quote_relation(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def _combine_residual(left: Optional[Expr], right: Optional[Expr]) -> Optional[Expr]:
    if left is None:
        return right
    if right is None:
        return left
    return left & right


def _column_type(expr: Expr, schema: pa.Schema) -> Optional[Tuple[str, pa.DataType]]:
    if isinstance(expr, AliasExpr):
        expr = expr.expr
    if not isinstance(expr, ColumnExpr):
        return None
    index = schema.get_field_index(expr.name)
    if index < 0:
        return None
    return expr.name, schema.field(index).type


def _literal_sql(value, data_type: pa.DataType) -> Optional[str]:
    if value is None:
        return "NULL"
    if pa.types.is_boolean(data_type):
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        return None
    if pa.types.is_integer(data_type):
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return str(value)
    if pa.types.is_floating(data_type):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        return repr(float(value))
    if pa.types.is_decimal(data_type):
        if not isinstance(value, (Decimal, int)) or isinstance(value, bool):
            return None
        decimal_value = Decimal(value)
        if not decimal_value.is_finite():
            return None
        return format(decimal_value, "f")
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        if not isinstance(value, str):
            return None
        if "'" in value or "\\" in value or any(ord(char) < 32 for char in value):
            return None
        return "'" + value + "'"
    if pa.types.is_date(data_type):
        if isinstance(value, datetime.datetime):
            return None
        if isinstance(value, datetime.date):
            return f"DATE '{value.isoformat()}'"
        return None
    if pa.types.is_timestamp(data_type):
        if not isinstance(value, datetime.datetime) or value.tzinfo is not None:
            return None
        stamp = value.isoformat(sep=" ")
        return f"TIMESTAMP '{stamp}'"
    return None


def _compile_exact(expr: Expr, schema: pa.Schema) -> Optional[str]:
    if isinstance(expr, AliasExpr):
        return _compile_exact(expr.expr, schema)

    if isinstance(expr, BinaryExpr):
        if expr.op in (Operation.AND, Operation.OR):
            left = _compile_exact(expr.left, schema)
            right = _compile_exact(expr.right, schema)
            if left is None or right is None:
                return None
            op = "AND" if expr.op == Operation.AND else "OR"
            return f"({left}) {op} ({right})"

        if expr.op in (Operation.IN, Operation.NOT_IN):
            column = _column_type(expr.left, schema)
            if column is None or not isinstance(expr.right, LiteralExpr):
                return None
            values = expr.right.value
            if not isinstance(values, (list, tuple)) or not values:
                return None
            if len(values) > _MAX_IN_LIST_VALUES or any(
                value is None for value in values
            ):
                return None
            rendered = [_literal_sql(value, column[1]) for value in values]
            if any(value is None for value in rendered):
                return None
            operator = "NOT IN" if expr.op == Operation.NOT_IN else "IN"
            return f"{quote_identifier(column[0])} {operator} ({', '.join(rendered)})"

        comparisons = {
            Operation.EQ: "=",
            Operation.NE: "<>",
            Operation.GT: ">",
            Operation.GE: ">=",
            Operation.LT: "<",
            Operation.LE: "<=",
        }
        operator = comparisons.get(expr.op)
        if operator is None:
            return None

        left_column = _column_type(expr.left, schema)
        right_column = _column_type(expr.right, schema)
        if left_column is not None and isinstance(expr.right, LiteralExpr):
            column, literal = left_column, expr.right.value
        elif right_column is not None and isinstance(expr.left, LiteralExpr):
            column, literal = right_column, expr.left.value
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(
                operator, operator
            )
        else:
            return None

        if literal is None:
            # Ray's nullable comparison expressions retain null results, while
            # SQL IS NULL has a different boolean result. Leave this to Ray.
            return None

        rendered = _literal_sql(literal, column[1])
        if rendered is None:
            return None
        return f"{quote_identifier(column[0])} {operator} {rendered}"

    if isinstance(expr, UnaryExpr):
        if expr.op in (Operation.IS_NULL, Operation.IS_NOT_NULL):
            column = _column_type(expr.operand, schema)
            if column is None:
                return None
            operator = "IS NULL" if expr.op == Operation.IS_NULL else "IS NOT NULL"
            return f"{quote_identifier(column[0])} {operator}"
        if expr.op == Operation.NOT:
            operand = _compile_exact(expr.operand, schema)
            return f"NOT ({operand})" if operand is not None else None

    return None


def split_predicate(
    predicate: Expr, schema: pa.Schema
) -> Tuple[Optional[str], Optional[Expr]]:
    """Compile safe conjuncts and preserve every unsupported residual."""
    if isinstance(predicate, BinaryExpr) and predicate.op == Operation.AND:
        left_sql, left_residual = split_predicate(predicate.left, schema)
        right_sql, right_residual = split_predicate(predicate.right, schema)
        if left_sql is None and right_sql is None:
            return None, _combine_residual(left_residual, right_residual)
        pushed_sql = " AND ".join(
            f"({sql})" for sql in (left_sql, right_sql) if sql is not None
        )
        return pushed_sql, _combine_residual(left_residual, right_residual)

    sql = _compile_exact(predicate, schema)
    return (sql, None) if sql is not None else (None, predicate)
