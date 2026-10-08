import copy
from functools import partial
from typing import TYPE_CHECKING, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.dataset as pds
from pyarrow.fs import LocalFileSystem
from typing_extensions import override

from ray.data._internal.arrow_block import _BATCH_SIZE_PRESERVING_STUB_COL_NAME
from ray.data._internal.datasource_v2.common.file_reader import FileReader
from ray.data._internal.datasource_v2.formats.orc.orc_metadata import (
    orc_read_unit_id,
    read_orc_metadata,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data._internal.datasource_v2.interfaces.read_units import (
    ReadUnit,
    ReadUnitFragment,
)
from ray.data._internal.datasource_v2.interfaces.synthesized_columns import (
    ReadUnitPosition,
)
from ray.data._internal.object_extensions.arrow import raise_on_pickle_object_columns
from ray.data._internal.planner.plan_expression.expression_visitors import (
    get_column_references,
)
from ray.data._internal.util import iterate_with_retry
from ray.data.context import DataContext
from ray.data.datasource.file_based_datasource import _add_partitions_to_table
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from pyarrow.orc import ORCFile


@DeveloperAPI
class OrcFileReader(FileReader):
    """Read ORC files in batches using PyArrow Dataset fragments.

    Manifest chunks select physical stripes, decoded by PyArrow. Without chunk
    metadata, the existing whole-file scanner is used. Reading a stripe
    materializes it before yielding batches, so batch_size is not a peak-memory
    bound.
    Partitioned reads validate and synthesize partition values before projection.
    """

    @override
    def read(self, input_split: FileManifest) -> Iterator[pa.Table]:
        """Project partitioned reads after validating and synthesizing columns."""
        reader = self
        if self._columns is not None and self._partition_parser is not None:
            reader = copy.copy(self)
            reader._columns = None

        for table in FileReader.read(reader, input_split):
            if reader is not self:
                assert self._columns is not None
                produced = set(table.column_names)
                table = table.select(
                    [name for name in self._columns if name in produced]
                )
                if table.num_columns == 0 and table.num_rows > 0:
                    table = table.append_column(
                        _BATCH_SIZE_PRESERVING_STUB_COL_NAME,
                        pa.nulls(table.num_rows),
                    )
            elif self._columns is None and self._schema is not None:
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
    def _get_fragments_to_read(
        self, dataset: pds.Dataset, manifest: FileManifest
    ) -> List[ReadUnitFragment]:
        if all(md is None for md in manifest.file_chunk_metadatas):
            return super()._get_fragments_to_read(dataset, manifest)
        by_path = {fragment.path: fragment for fragment in dataset.get_fragments()}
        layouts = {}
        result = []
        for path, chunk in zip(manifest.paths, manifest.file_chunk_metadatas):
            fragment = by_path[path]
            if chunk is None:
                result.append(
                    ReadUnitFragment(fragment, ReadUnit(id=path, source=path, count=1))
                )
                continue
            if path not in layouts:
                layouts[path] = read_orc_metadata(path, self._filesystem)
            layout = layouts[path]
            if layout is None:
                raise ModuleNotFoundError(
                    "Reading ORC stripe manifests requires pyorc on every read worker."
                )
            indices = [int(i) for i in chunk["unit_ids"]]
            if indices == list(range(len(layout.stripes))) and not any(
                column.requires_read_unit_boundaries
                for column in self._synthesized_columns
            ):
                # Coalescing all stripes back into one task needs no stripe
                # materialization. Keep the existing streamed whole-file path.
                result.append(
                    ReadUnitFragment(
                        fragment,
                        ReadUnit(
                            id=path, source=path, count=1, num_rows=layout.num_rows
                        ),
                    )
                )
                continue
            for index in indices:
                if not 0 <= index < len(layout.stripes):
                    raise ValueError(f"Invalid ORC stripe {index} for {path}")
                stripe = layout.stripes[index]
                result.append(
                    ReadUnitFragment(
                        fragment,
                        ReadUnit(
                            id=orc_read_unit_id(path, index),
                            source=path,
                            index=index,
                            count=len(layout.stripes),
                            num_rows=stripe.num_rows,
                        ),
                        unit_start_row=stripe.row_offset,
                    )
                )
        return result

    @override
    def _read_fragments_sequential(
        self,
        fragments_with_offsets: Iterator[ReadUnitFragment],
        scanner_kwargs: dict,
    ) -> Iterator[Tuple[pa.Table, ReadUnitPosition]]:
        for item in fragments_with_offsets:
            if item.unit.id == item.fragment.path:
                yield from super()._read_fragments_sequential(
                    iter([item]), scanner_kwargs
                )
                continue
            rows_before = 0
            for table in iterate_with_retry(
                partial(
                    self._iter_stripe_tables,
                    item.fragment,
                    item.unit.index,
                    scanner_kwargs,
                ),
                f"read ORC stripe {item.unit.index} from {item.fragment.path}",
                match=DataContext.get_current().retried_io_errors,
            ):
                if table.num_rows:
                    yield table, ReadUnitPosition(
                        unit=item.unit,
                        rows_before=rows_before,
                        unit_start_row=item.unit_start_row,
                    )
                    rows_before += table.num_rows

    def _iter_stripe_tables(
        self, fragment: pds.Fragment, stripe_index: int, scanner_kwargs: dict
    ) -> Iterator[pa.Table]:
        import pyarrow.orc as orc

        filesystem = self._filesystem or LocalFileSystem()
        with filesystem.open_input_file(fragment.path) as source:
            yield from self._scan_stripe(
                orc.ORCFile(source), fragment.path, stripe_index, scanner_kwargs
            )

    def _scan_stripe(
        self,
        orc_file: "ORCFile",
        path: str,
        stripe_index: int,
        scanner_kwargs: dict,
    ) -> Iterator[pa.Table]:
        physical_schema = orc_file.schema
        schema = self._schema if self._schema is not None else physical_schema
        synthesized = {column.name for column in self._synthesized_columns}
        schema = pa.schema([field for field in schema if field.name not in synthesized])
        partitions = self._partition_parser(path) if self._partition_parser else {}
        for name in partitions:
            physical_index = physical_schema.get_field_index(name)
            if physical_index != -1:
                field = physical_schema.field(physical_index)
                index = schema.get_field_index(name)
                schema = (
                    schema.append(field) if index == -1 else schema.set(index, field)
                )
        requested = scanner_kwargs.get("columns")
        names = list(schema.names if requested is None else requested)
        if self._predicate is not None:
            names.extend(get_column_references(self._predicate))
        names.extend(partitions)
        columns = list(
            dict.fromkeys(
                name for name in names if physical_schema.get_field_index(name) != -1
            )
        )
        # Read one physical column to preserve row counts for an empty projection.
        if not columns:
            columns = physical_schema.names[:1]
        table = pa.Table.from_batches(
            [orc_file.read_stripe(stripe_index, columns=columns)]
        )
        if not table.num_rows:
            return
        raise_on_pickle_object_columns(table)
        if partitions:
            table = _add_partitions_to_table(table, partitions)
        for fragment in pds.dataset(table).get_fragments():
            scanner = fragment.scanner(**scanner_kwargs, schema=schema)
            for tagged in scanner.scan_batches():
                yield pa.Table.from_batches([tagged.record_batch])

    @override
    def _iter_fragment_tables(
        self, fragment: pds.Fragment, scanner_kwargs: dict
    ) -> Iterator[pa.Table]:
        if self._partition_parser is None:
            yield from super()._iter_fragment_tables(fragment, scanner_kwargs)
            return

        partitions = self._partition_parser(fragment.path)
        physical_schema = fragment.physical_schema
        schema = self._schema if self._schema is not None else physical_schema
        synthesized = {column.name for column in self._synthesized_columns}
        schema = pa.schema([field for field in schema if field.name not in synthesized])
        # Keep real partition columns long enough to enforce V1's consistency
        # check. A missing column is only an Arrow null-fill placeholder.
        for name in partitions:
            index = physical_schema.get_field_index(name)
            if index != -1:
                field = physical_schema.field(index)
                index = schema.get_field_index(name)
                schema = (
                    schema.append(field) if index == -1 else schema.set(index, field)
                )

        if any(physical_schema.get_field_index(name) != -1 for name in partitions):
            import pyarrow.orc as orc

            # V1 validates a whole stripe. Validating individual scan batches
            # would reject an all-null batch within an otherwise valid stripe.
            filesystem = self._filesystem or LocalFileSystem()
            columns = [
                name
                for name in schema.names
                if physical_schema.get_field_index(name) != -1
            ]
            with filesystem.open_input_file(fragment.path) as source:
                orc_file = orc.ORCFile(source)
                for stripe_index in range(orc_file.nstripes):
                    table = pa.Table.from_batches(
                        [orc_file.read_stripe(stripe_index, columns=columns)]
                    )
                    if table.num_rows == 0:
                        continue
                    raise_on_pickle_object_columns(table)
                    table = _add_partitions_to_table(table, partitions)
                    # Reuse Arrow's schema alignment and batch sizing after
                    # validation, including null-fill for missing data fields.
                    for stripe in pds.dataset(table).get_fragments():
                        scanner = stripe.scanner(**scanner_kwargs, schema=schema)
                        for tagged in scanner.scan_batches():
                            yield pa.Table.from_batches([tagged.record_batch])
            return

        scanner = fragment.scanner(**scanner_kwargs, schema=schema)
        for tagged in scanner.scan_batches():
            yield pa.Table.from_batches([tagged.record_batch])

    @override
    def _on_batch_read(self, table: pa.Table) -> None:
        super()._on_batch_read(table)
        raise_on_pickle_object_columns(table)
