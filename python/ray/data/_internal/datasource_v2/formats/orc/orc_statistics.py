"""Conservative ORC statistics pruning; row filters remain in the reader."""

import operator
from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, overload

import numpy as np
from typing_extensions import override

from ray.data._internal.datasource_v2.formats.orc.orc_metadata import (
    OrcFileMetadata,
    OrcStripeMetadata,
    read_orc_metadata,
)
from ray.data._internal.datasource_v2.formats.orc.orc_stripe_indexer import (
    OrcStripeIndexer,
)
from ray.data._internal.planner.plan_expression.expression_visitors import (
    get_column_references,
)
from ray.data.expressions import BinaryExpr, ColumnExpr, Expr, LiteralExpr, Operation
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem

_REVERSE = {
    Operation.EQ: Operation.EQ,
    Operation.NE: Operation.NE,
    Operation.LT: Operation.GT,
    Operation.LE: Operation.GE,
    Operation.GT: Operation.LT,
    Operation.GE: Operation.LE,
}
_INTEGER_KINDS = {"BYTE", "SHORT", "INT", "LONG"}


@overload
def _bounds_can_match(
    op: Operation, low: int, high: int, value_low: int, value_high: int
) -> bool:
    ...


@overload
def _bounds_can_match(
    op: Operation, low: str, high: str, value_low: str, value_high: str
) -> bool:
    ...


def _bounds_can_match(
    op: Operation,
    low: int | str,
    high: int | str,
    value_low: int | str,
    value_high: int | str,
) -> bool:
    if op == Operation.EQ:
        return bool(operator.le(low, value_high) and operator.ge(high, value_low))
    if op == Operation.NE:
        return not (low == high == value_low == value_high)
    if op == Operation.LT:
        return bool(operator.lt(low, value_high))
    if op == Operation.LE:
        return bool(operator.le(low, value_high))
    if op == Operation.GT:
        return bool(operator.gt(high, value_low))
    return bool(operator.ge(high, value_low))


def _integer_bounds(low: int, high: int) -> Tuple[int, int]:
    # The listing API has physical statistics but no unified scan schema.
    # Include float32/float64 rounding so an integer promoted to a floating
    # schema cannot be incorrectly pruned. ORC has no float16 physical type.
    lows = (low, int(np.float32(low)), int(float(low)))
    highs = (high, int(np.float32(high)), int(float(high)))
    return min(lows), max(highs)


def statistics_can_match(
    predicate: Optional[Expr], statistics: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Keep a unit unless supported statistics prove that no row can match."""
    if not isinstance(predicate, BinaryExpr):
        return True
    if predicate.op == Operation.AND:
        return statistics_can_match(
            predicate.left, statistics
        ) and statistics_can_match(predicate.right, statistics)
    if predicate.op == Operation.OR:
        return statistics_can_match(predicate.left, statistics) or statistics_can_match(
            predicate.right, statistics
        )
    op = predicate.op
    left, right = predicate.left, predicate.right
    if isinstance(left, LiteralExpr) and isinstance(right, ColumnExpr):
        left, right = right, left
        op = _REVERSE.get(op, op)
    if (
        op not in _REVERSE
        or not isinstance(left, ColumnExpr)
        or not isinstance(right, LiteralExpr)
    ):
        return True
    stats = statistics.get(left.name)
    if stats is None:
        return True
    kind = getattr(stats.get("kind"), "name", None)
    value = right.value
    if kind in _INTEGER_KINDS:
        if not isinstance(value, Integral) or isinstance(value, bool):
            return True
        value = int(value)
        # Arrow cannot represent an arbitrary Python bigint as an int64 scalar.
        if not -(1 << 63) <= value < (1 << 63):
            return True
    elif kind == "STRING":
        if not isinstance(value, str):
            return True
        value = str(value)
    else:
        # Float statistics can omit NaNs; other types need their own semantics.
        return True
    if stats.get("number_of_values") == 0:
        return False
    low, high = stats.get("minimum"), stats.get("maximum")
    if low is None or high is None:
        return True
    if kind in _INTEGER_KINDS:
        if (
            not isinstance(low, Integral)
            or not isinstance(high, Integral)
            or isinstance(low, bool)
            or isinstance(high, bool)
            or low > high
        ):
            return True
        integer_low, integer_high = _integer_bounds(int(low), int(high))
        value_low, value_high = _integer_bounds(int(value), int(value))
        return _bounds_can_match(op, integer_low, integer_high, value_low, value_high)
    else:
        if not isinstance(low, str) or not isinstance(high, str) or low > high:
            return True
        return _bounds_can_match(op, str(low), str(high), str(value), str(value))


@DeveloperAPI
class OrcStatisticsIndexer(OrcStripeIndexer):
    """Prune file and stripe statistics while retaining exact row filtering."""

    def __init__(self, *, ignore_missing_paths: bool, enable_pruning: bool):
        super().__init__(ignore_missing_paths=ignore_missing_paths)
        self._enable_pruning = enable_pruning

    @override
    def _read_metadata(
        self,
        path: str,
        filesystem: Optional["FileSystem"],
        predicate: Optional[Expr],
    ) -> Optional[OrcFileMetadata]:
        columns = (
            set(get_column_references(predicate))
            if predicate is not None and self._enable_pruning
            else None
        )
        return read_orc_metadata(path, filesystem, statistic_columns=columns)

    @override
    def _file_can_match(
        self, metadata: OrcFileMetadata, predicate: Optional[Expr]
    ) -> bool:
        return not self._enable_pruning or statistics_can_match(
            predicate, metadata.statistics
        )

    @override
    def _stripe_can_match(
        self, stripe: OrcStripeMetadata, predicate: Optional[Expr]
    ) -> bool:
        return not self._enable_pruning or statistics_can_match(
            predicate, stripe.statistics
        )
