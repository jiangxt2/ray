import math
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

import pyarrow as pa
import pyarrow.dataset as pds
from pyarrow.fs import FileSystem
from typing_extensions import override

from ray.data._internal.datasource_v2.common.file_reader import (
    _ARROW_DEFAULT_BATCH_SIZE,
    FileFormat,
    FileReader,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data._internal.datasource_v2.interfaces.synthesized_columns import (
    SynthesizedColumn,
)
from ray.data._internal.object_extensions.arrow import raise_on_pickle_object_columns
from ray.data._internal.util import call_with_retry
from ray.data.context import DataContext
from ray.data.expressions import Expr
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from ray.data.datasource.partitioning import Partitioning


@DeveloperAPI
class OrcFileReader(FileReader):
    """Read ORC files in batches using PyArrow Dataset fragments.

    Each fragment covers a whole file. PyArrow applies row filters to the
    scanned batches; this reader does not provide ORC-native stripe pruning.
    """

    _BATCH_SIZE_SAMPLE_ROWS = 1024

    def __init__(
        self,
        format: FileFormat = FileFormat.ORC,
        batch_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        predicate: Optional[Expr] = None,
        limit: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        partitioning: "Optional[Partitioning]" = None,
        ignore_prefixes: Optional[List[str]] = None,
        synthesized_columns: Sequence[SynthesizedColumn] = (),
        schema: Optional[pa.Schema] = None,
        target_block_size: Optional[int] = None,
    ):
        """Initialize the ORC reader.

        Args:
            format: File format passed to the base reader.
            batch_size: Explicit row count, which takes priority over estimates.
            columns: Columns to read, or None for all columns.
            predicate: Row filter applied by PyArrow.
            limit: Maximum number of output rows.
            filesystem: Filesystem used to open files.
            partitioning: Path-derived partition columns.
            ignore_prefixes: File name prefixes ignored by PyArrow.
            synthesized_columns: Columns appended after reading.
            schema: Unified schema for file reads and partition values.
            target_block_size: Target batch size in bytes when batch_size is unset.
        """
        super().__init__(
            format=format,
            batch_size=(
                batch_size if batch_size is not None else _ARROW_DEFAULT_BATCH_SIZE
            ),
            columns=columns,
            predicate=predicate,
            limit=limit,
            filesystem=filesystem,
            partitioning=partitioning,
            ignore_prefixes=ignore_prefixes,
            synthesized_columns=synthesized_columns,
            schema=schema,
        )
        self._explicit_batch_size = batch_size
        self._target_block_size = target_block_size
        self._sampled_batch_size: Optional[int] = None

    @override
    def _resolve_batch_size(self, dataset: pds.Dataset, manifest: FileManifest) -> int:
        if self._explicit_batch_size is not None:
            return self._explicit_batch_size
        if self._sampled_batch_size is not None:
            return self._sampled_batch_size

        batch_size = _ARROW_DEFAULT_BATCH_SIZE
        if self._target_block_size is not None:
            estimated = call_with_retry(
                lambda: self._estimate_batch_size(dataset),
                description="sample ORC batch size",
                match=DataContext.get_current().retried_io_errors,
            )
            if estimated is not None:
                batch_size = estimated

        self._sampled_batch_size = batch_size
        return batch_size

    def _estimate_batch_size(self, dataset: pds.Dataset) -> Optional[int]:
        columns = (
            [name for name in self._columns if name in dataset.schema.names]
            if self._columns is not None
            else None
        )
        # ORC has no Python row-group size statistics. Sample a small batch
        # without a row filter, so a selective predicate cannot hide its cost.
        batches = dataset.scanner(
            columns=columns,
            batch_size=self._BATCH_SIZE_SAMPLE_ROWS,
            batch_readahead=0,
            fragment_readahead=0,
        ).to_reader()
        try:
            batch = next(batches, None)
            if batch is None or batch.num_rows == 0 or batch.nbytes == 0:
                return None
            return self._batch_size_from_row_size(batch.nbytes / batch.num_rows)
        finally:
            batches.close()

    def _batch_size_from_row_size(self, row_size: float) -> int:
        assert self._target_block_size is not None
        # Keep the default row count as an upper bound, including when the
        # context disables block sizing with an effectively infinite target.
        return max(
            1,
            min(
                math.ceil(self._target_block_size / row_size), _ARROW_DEFAULT_BATCH_SIZE
            ),
        )

    @override
    def read(self, input_split: FileManifest) -> Iterator[pa.Table]:
        """Keep the declared column order after synthesized fields are appended."""
        for table in super().read(input_split):
            if self._columns is None and self._schema is not None:
                produced = set(table.column_names)
                schema_names = self._schema.names
                schema_name_set = set(schema_names)
                column_names = [name for name in schema_names if name in produced]
                column_names.extend(
                    name for name in table.column_names if name not in schema_name_set
                )
                table = table.select(column_names)
            yield table

    @override
    def _make_format(self) -> pds.OrcFileFormat:
        return pds.OrcFileFormat()

    @override
    def _on_batch_read(self, table: pa.Table) -> None:
        super()._on_batch_read(table)
        raise_on_pickle_object_columns(table)
        if self._target_block_size is None or table.num_rows == 0 or table.nbytes == 0:
            return
        self._sampled_batch_size = self._batch_size_from_row_size(
            table.nbytes / table.num_rows
        )
