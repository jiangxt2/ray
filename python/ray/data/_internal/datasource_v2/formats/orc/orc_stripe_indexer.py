"""ORC stripe listing and grouping using shared DSV2 contracts."""

import math
from collections import deque
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Deque,
    Iterable,
    List,
    Optional,
)

from typing_extensions import override

from ray.data._internal.datasource_v2.common.non_sampling_file_indexer import (
    NonSamplingFileIndexer,
)
from ray.data._internal.datasource_v2.common.online_bin_packer import OnlineBinPacker
from ray.data._internal.datasource_v2.formats.orc.orc_metadata import (
    OrcFileMetadata,
    OrcStripeMetadata,
    orc_read_unit_id,
    read_orc_metadata,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import (
    FileChunk,
    FileManifest,
)
from ray.data._internal.datasource_v2.interfaces.file_partitioner import (
    FilePartitioner,
    PartitionHints,
)
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem

    from ray.data._internal.datasource_v2.interfaces.file_pruner import FilePruner
    from ray.data.block import BlockColumn
    from ray.data.datasource.file_based_datasource import FileShuffleConfig
    from ray.data.expressions import Expr


@DeveloperAPI
class OrcStripeIndexer(NonSamplingFileIndexer):
    """List physical stripes when PyORC is available, otherwise whole files."""

    def _read_metadata(
        self, path: str, filesystem: Optional["FileSystem"], predicate: Optional["Expr"]
    ) -> Optional[OrcFileMetadata]:
        return read_orc_metadata(path, filesystem)

    def _file_can_match(
        self, metadata: OrcFileMetadata, predicate: Optional["Expr"]
    ) -> bool:
        return True

    def _stripe_can_match(
        self, stripe: OrcStripeMetadata, predicate: Optional["Expr"]
    ) -> bool:
        return True

    @override
    def list_files(
        self,
        paths: "BlockColumn",
        *,
        filesystem: Optional["FileSystem"],
        pruners: Optional[List["FilePruner"]] = None,
        preserve_order: bool = False,
        predicate: Optional["Expr"] = None,
        limit: Optional[int] = None,
        projected_columns: Optional[List[str]] = None,
        shuffle_config: Optional["FileShuffleConfig"] = None,
        execution_idx: int = 0,
        excluded_read_unit_ids: Optional[AbstractSet[str]] = None,
    ) -> Iterable[FileManifest]:
        files = self._iter_file_infos_for_list(
            paths,
            filesystem=filesystem,
            pruners=pruners,
            preserve_order=preserve_order,
            shuffle_config=shuffle_config,
            execution_idx=execution_idx,
            excluded_read_unit_ids=excluded_read_unit_ids,
        )
        matched_rows = 0
        seen_paths = set()
        for file in files:
            if file.size is None or file.path in seen_paths:
                continue
            seen_paths.add(file.path)
            metadata = self._read_metadata(file.path, filesystem, predicate)
            if metadata is None:
                yield from self._process_file_infos_to_manifests([file])
                continue
            if not self._file_can_match(metadata, predicate):
                continue
            for stripe in metadata.stripes:
                if (
                    stripe.num_rows == 0
                    or orc_read_unit_id(file.path, stripe.index)
                    in (excluded_read_unit_ids or ())
                    or not self._stripe_can_match(stripe, predicate)
                ):
                    continue
                # With a predicate, statistics only establish possibility.
                # Never count the physical rows as exact survivors for limit.
                chunk = FileChunk(
                    unit_ids=(stripe.index,),
                    num_rows=stripe.num_rows,
                    size_bytes=stripe.size_bytes,
                    fully_matched=predicate is None,
                )
                assert file.size is not None
                yield FileManifest.construct_manifest(
                    paths=[file.path],
                    sizes=[file.size],
                    chunk_metadatas=[chunk.to_metadata()],
                )
                if predicate is None:
                    matched_rows += stripe.num_rows
                    if limit is not None and matched_rows >= limit:
                        return


@DeveloperAPI
class OrcStripePartitioner(FilePartitioner):
    """Use shared bin packing with a budget derived from requested parallelism."""

    def __init__(self, hints: PartitionHints):
        self._hints = hints
        self._preserve_order = (
            DataContext.get_current().execution_options.preserve_order
        )
        self._inputs: List[FileManifest] = []
        self._output: Deque[FileManifest] = deque()

    @property
    @override
    def requires_global_input(self) -> bool:
        return True

    @override
    def add_input(self, input_manifest: FileManifest) -> None:
        self._inputs.append(input_manifest)

    @override
    def has_partition(self) -> bool:
        return bool(self._output)

    @override
    def next_partition(self) -> FileManifest:
        return self._output.popleft()

    @override
    def finalize(self) -> None:
        total_bytes = sum(
            int(metadata["size_bytes"]) if metadata is not None else int(size)
            for manifest in self._inputs
            for size, metadata in zip(
                manifest.file_sizes, manifest.file_chunk_metadatas
            )
        )
        # These are encoded stripe bytes, not a decoded-memory guarantee.
        budget = max(
            1,
            min(
                self._hints.max_bucket_size,
                max(
                    self._hints.min_bucket_size,
                    math.ceil(total_bytes / max(1, self._hints.num_buckets)),
                ),
            ),
        )
        if self._preserve_order:
            bucket = []
            bucket_bytes = 0
            for manifest in self._inputs:
                for path, size, metadata in zip(
                    manifest.paths, manifest.file_sizes, manifest.file_chunk_metadatas
                ):
                    weight = (
                        int(metadata["size_bytes"])
                        if metadata is not None
                        else int(size)
                    )
                    if bucket and bucket_bytes + weight > budget:
                        self._output.append(FileManifest.concat(bucket))
                        bucket = []
                        bucket_bytes = 0
                    bucket.append(
                        FileManifest.construct_manifest(
                            paths=[str(path)],
                            sizes=[int(size)],
                            chunk_metadatas=[metadata],
                        )
                    )
                    bucket_bytes += weight
            if bucket:
                self._output.append(FileManifest.concat(bucket))
            self._inputs.clear()
            return
        packer = OnlineBinPacker(max_bin_bytes=budget, split_coalesced=True)
        for manifest in self._inputs:
            packer.add_input(manifest)
        packer.finalize()
        while packer.has_partition():
            self._output.append(packer.next_partition())
        self._inputs.clear()
