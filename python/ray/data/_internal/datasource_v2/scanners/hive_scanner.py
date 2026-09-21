"""Immutable scanner configuration for table-oriented HiveServer2 reads."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import pyarrow as pa
from typing_extensions import override

from ray.data._internal.datasource_v2.hive_sql import (
    quote_identifier,
    split_predicate,
)
from ray.data._internal.datasource_v2.logical_optimizers import (
    SupportsColumnPruning,
    SupportsFilterPushdown,
    SupportsLimitPushdown,
)
from ray.data._internal.datasource_v2.readers.hive_reader import (
    DEFAULT_HIVE_TARGET_BATCH_BYTES,
    HiveConnectionOptions,
    HiveReader,
)
from ray.data._internal.datasource_v2.scanners.scanner import Scanner
from ray.data.expressions import Expr


@dataclass(frozen=True)
class HiveScanner(
    Scanner,
    SupportsFilterPushdown,
    SupportsColumnPruning,
    SupportsLimitPushdown,
):
    schema: pa.Schema
    connection_options: HiveConnectionOptions = field(repr=False)
    base_query: str = field(repr=False)
    table_mode: bool
    source_limit: Optional[int] = None
    columns: Optional[Tuple[str, ...]] = None
    predicate_sql: Optional[str] = field(default=None, repr=False)
    predicate: Optional[Expr] = field(default=None, repr=False)
    limit: Optional[int] = None
    target_batch_bytes: int = DEFAULT_HIVE_TARGET_BATCH_BYTES

    @override
    def read_schema(self) -> pa.Schema:
        if self.columns is None or len(self.columns) == 0:
            return self.schema
        try:
            return self.schema.select(list(self.columns))
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(
                f"Hive projection references unknown column {missing!r}"
            ) from None

    @override
    def create_reader(self) -> HiveReader:
        query = self._build_query()
        if self.table_mode and self.columns:
            query_schema = self.schema.select(list(self.columns))
            local_output_columns = None
        else:
            query_schema = self.schema
            local_output_columns = (
                self.columns
                if not self.table_mode and self.columns is not None
                else None
            )
        return HiveReader(
            connection_options=self.connection_options,
            query=query,
            expected_query_schema=query_schema,
            output_schema=self.read_schema(),
            local_output_columns=local_output_columns,
            target_batch_bytes=self.target_batch_bytes,
        )

    @override
    def prune_columns(self, columns: List[str]) -> "HiveScanner":
        if not columns:
            # Keep the input schema intact for Ray's row-count-preserving
            # projection operator when users select zero columns.
            return replace(self, columns=None)
        unknown = [name for name in columns if self.schema.get_field_index(name) < 0]
        if unknown:
            raise ValueError(f"Hive projection references unknown columns: {unknown!r}")
        return replace(self, columns=tuple(columns))

    @override
    def pruned_column_names(self) -> Optional[Tuple[str, ...]]:
        return self.columns

    @override
    def push_filters(self, predicate: Expr):
        if not self.table_mode:
            return self, predicate

        pushed_sql, residual = split_predicate(predicate, self.schema)
        if pushed_sql is None:
            return self, predicate

        combined_sql = (
            f"({self.predicate_sql}) AND ({pushed_sql})"
            if self.predicate_sql
            else pushed_sql
        )
        combined_predicate = (
            self.predicate & predicate if self.predicate is not None else predicate
        )
        return (
            replace(
                self,
                predicate_sql=combined_sql,
                predicate=combined_predicate,
            ),
            residual,
        )

    @override
    def push_limit(self, limit: int) -> "HiveScanner":
        if not self.table_mode:
            return self
        new_limit = limit if self.limit is None else min(self.limit, limit)
        return replace(self, limit=new_limit)

    @override
    def pushed_limit(self) -> Optional[int]:
        if not self.table_mode:
            return None
        return self.limit

    def _build_query(self) -> str:
        if not self.table_mode:
            return self.base_query

        selected_columns = self.columns or tuple(self.schema.names)
        select_clause = ", ".join(
            f"{quote_identifier(name)} AS {quote_identifier(name)}"
            for name in selected_columns
        )
        query = f"SELECT {select_clause} FROM {self.base_query}"
        if self.predicate_sql:
            query += f" WHERE {self.predicate_sql}"
        limits = [
            value for value in (self.source_limit, self.limit) if value is not None
        ]
        if limits:
            query += f" LIMIT {min(limits)}"
        return query
