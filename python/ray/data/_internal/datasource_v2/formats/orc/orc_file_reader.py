from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Set

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.orc as orc
from pyarrow.fs import LocalFileSystem
from typing_extensions import override

from ray._common.utils import env_integer
from ray.data._internal.datasource_v2.common.file_reader import FileReader
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data._internal.datasource_v2.interfaces.supports_metadata import (
    MetadataType,
    SupportsMetadata,
)
from ray.data._internal.object_extensions.arrow import raise_on_pickle_object_columns
from ray.data._internal.util import call_with_retry
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class OrcFileReader(FileReader, SupportsMetadata):
    """Read ORC files in batches using PyArrow Dataset fragments.

    Each fragment covers a whole file. PyArrow applies row filters to the
    scanned batches; this reader does not provide ORC-native stripe pruning.
    """

    _COUNT_ROWS_BATCH_SIZE = env_integer(
        "RAY_DATA_ORC_READER_COUNT_ROWS_BATCH_SIZE", 16
    )

    @override
    def read_metadata(self, file_manifest: FileManifest) -> Iterator[BlockMetadata]:
        """Yield per-file row counts without decoding ORC data columns."""
        filesystem = self._filesystem or LocalFileSystem()
        retried_io_errors = DataContext.get_current().retried_io_errors

        def read_num_rows(path: str) -> int:
            def read_footer() -> int:
                with filesystem.open_input_file(path) as source:
                    try:
                        return orc.ORCFile(source).nrows
                    except (OSError, pa.ArrowInvalid) as error:
                        raise type(error)(
                            f"Failed to read ORC footer for {path}: {error}"
                        ) from error

            return call_with_retry(
                read_footer,
                description=f"read ORC footer for {path}",
                match=retried_io_errors,
            )

        with ThreadPoolExecutor() as executor:
            for num_rows in executor.map(read_num_rows, map(str, file_manifest.paths)):
                yield BlockMetadata(
                    num_rows=num_rows,
                    size_bytes=None,
                    exec_stats=None,
                    input_files=None,
                )

    @override
    def available_metadata(self) -> Set[MetadataType]:
        if self._predicate is not None:
            return set()
        return {MetadataType.NUM_ROWS}

    @override
    def get_target_metadata_batch_size(self) -> Optional[int]:
        return self._COUNT_ROWS_BATCH_SIZE

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
