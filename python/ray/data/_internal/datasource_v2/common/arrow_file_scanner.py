from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import pyarrow as pa
from pyarrow.fs import FileSystem
from typing_extensions import override

from ray.data._internal.datasource_v2.common.file_scanner import FileScanner
from ray.data._internal.datasource_v2.interfaces.pushdown import (
    SupportsColumnPruning,
    SupportsFilterPushdown,
    SupportsLimitPushdown,
)
from ray.data._internal.datasource_v2.interfaces.synthesized_columns import (
    SynthesizedColumn,
)
from ray.data.expressions import Expr
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class ArrowFileScanner(
    FileScanner,
    SupportsFilterPushdown,
    SupportsColumnPruning,
    SupportsLimitPushdown,
):
    """Base scanner for file-based datasources that use PyArrow's Dataset API.

    Holds shared Arrow types and options (schema, projection, filesystem,
    etc.). Subclasses set the file format in :meth:`create_reader`.

    Provides default implementations of filter pushdown, column pruning and
    limit pushdown that work for all Arrow-backed formats. Partition pruning
    comes from :class:`FileScanner`.

    Non-Arrow file formats should subclass :class:`FileScanner` directly.
    """

    schema: pa.Schema
    batch_size: Optional[int] = None
    columns: Optional[Tuple[str, ...]] = None
    predicate: Optional[Expr] = None
    limit: Optional[int] = None
    filesystem: Optional[FileSystem] = None
    ignore_prefixes: Optional[List[str]] = None

    @override
    def metadata_row_count_is_exact(self) -> bool:
        """``True`` when no row-reducing pushdown is set on this scanner.

        A Parquet footer's ``num_rows`` is the file's total, with nothing in it
        to say how many rows survive a filter, so for this scanner the question
        collapses to "is anything reducing rows?". Column projection is
        deliberately not consulted: it changes the width of the output, never
        the row count.
        """
        return (
            self.predicate is None
            and self.partition_predicate is None
            and self.limit is None
        )

    def read_schema(self) -> pa.Schema:
        """Return the logical schema after column pruning.

        ``columns is None`` → no projection applied, return the full schema.
        ``columns = ()`` → empty projection (``ds.select_columns([])``),
        return an empty schema.

        The physical read may still inject a stub column (see
        ``_BATCH_SIZE_PRESERVING_STUB_COL_NAME``) so that row counts
        survive a zero-column scan; that stub is an execution-layer detail
        and is deliberately not reflected in this logical schema.
        """
        return self._project_schema(self.schema)

    def _project_schema(self, schema: pa.Schema) -> pa.Schema:
        if self.columns is None:
            return schema
        fields = []
        for name in self.columns:
            idx = schema.get_field_index(name)
            assert idx >= 0, f"Column {name} not found in schema"
            fields.append(schema.field(idx))
        return pa.schema(fields)

    def _read_schema_with_synthesized_columns(
        self,
        synthesized_columns: Sequence[SynthesizedColumn],
        *,
        replace_existing: bool = False,
    ) -> pa.Schema:
        """Add synthesized fields while preserving each format's schema policy.

        Parquet projects the datasource-provided schema before adding missing
        fields. ORC also normalizes same-named fields to the synthesized type,
        and permits projecting a synthesized field absent from that schema.
        """
        schema = self.schema if replace_existing else self._project_schema(self.schema)
        for column in synthesized_columns:
            if self.columns is not None and column.name not in self.columns:
                continue
            field = pa.field(column.name, column.type)
            index = schema.get_field_index(column.name)
            if index == -1:
                schema = schema.append(field)
            elif replace_existing and (
                self.columns is not None or schema.field(index).type != column.type
            ):
                schema = schema.set(index, field)
        return self._project_schema(schema) if replace_existing else schema

    @override
    def push_filters(
        self, predicate: "Expr"
    ) -> Tuple["ArrowFileScanner", Optional["Expr"]]:
        """Push filter predicate down to the scanner.

        ANDs the predicate with any existing predicate. The Ray ``Expr`` is
        retained as the source of truth so the reader can introspect filter
        columns; conversion to a PyArrow expression happens at the
        scanner-kwargs boundary in :class:`FileReader`.

        This method handles data-column predicates only. Partition predicates
        should be pushed via :meth:`prune_partitions` instead; the optimizer
        is responsible for splitting them before calling either method.

        Args:
            predicate: Ray Data expression to push down.

        Returns:
            A pair ``(scanner, residual)`` where ``scanner`` has the predicate
            merged into its PyArrow filter. ``residual`` is ``None`` because
            PyArrow handles the full filter at scan time.
        """
        if self.predicate is not None:
            combined = self.predicate & predicate
        else:
            combined = predicate

        return replace(self, predicate=combined), None

    @override
    def pushed_predicate(self) -> Optional["Expr"]:
        return self.predicate

    @override
    def prune_columns(self, columns: List[str]) -> "ArrowFileScanner":
        """Prune to only the specified columns.

        Args:
            columns: List of column names to keep.

        Returns:
            New scanner with column pruning applied.
        """
        if self.columns:
            existing = set(self.columns)
            columns = [c for c in columns if c in existing]

        return replace(self, columns=tuple(columns))

    @override
    def pruned_column_names(self) -> Optional[Tuple[str, ...]]:
        return self.columns

    @override
    def push_limit(self, limit: int) -> "ArrowFileScanner":
        """Push row limit down to the scanner.

        Args:
            limit: Maximum number of rows to read.

        Returns:
            New scanner with limit applied.
        """
        current = self.limit
        new_limit = min(current, limit) if current is not None else limit
        return replace(self, limit=new_limit)

    @override
    def pushed_limit(self) -> Optional[int]:
        return self.limit
