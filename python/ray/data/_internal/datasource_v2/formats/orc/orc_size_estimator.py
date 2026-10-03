from collections.abc import Generator
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.orc as orc
from pyarrow.fs import FileSystem, LocalFileSystem

from ray.data._internal.datasource_v2.common.file_reader import FileFormat
from ray.data._internal.datasource_v2.formats.orc.orc_file_reader import OrcFileReader
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data._internal.datasource_v2.interfaces.in_memory_size_estimator import (
    InMemorySizeEstimator,
)
from ray.data._internal.util import call_with_retry
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class OrcInMemorySizeEstimator(InMemorySizeEstimator):
    """Estimate decoded ORC file sizes from a bounded sample and footer row count."""

    _SAMPLE_ROWS = 1024

    def __init__(self, filesystem: Optional[FileSystem] = None):
        self._filesystem = filesystem or LocalFileSystem()
        self._reader = OrcFileReader(
            format=FileFormat.ORC,
            filesystem=self._filesystem,
            batch_size=self._SAMPLE_ROWS,
        )
        self._encoding_ratio: Optional[float] = None

    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        assert np.all(manifest.file_sizes >= 0)
        if self._encoding_ratio is None:
            for path, file_size in zip(manifest.paths, manifest.file_sizes):
                if not file_size:
                    continue
                self._encoding_ratio = self._estimate_encoding_ratio(path, file_size)
                if self._encoding_ratio is not None:
                    break
        if self._encoding_ratio is None:
            return manifest.file_sizes
        return manifest.file_sizes * self._encoding_ratio

    def _estimate_encoding_ratio(self, path: str, file_size: int) -> Optional[float]:
        def read_num_rows() -> int:
            with self._filesystem.open_input_file(path) as source:
                try:
                    return orc.ORCFile(source).nrows
                except (OSError, pa.ArrowInvalid) as error:
                    raise type(error)(
                        f"Failed to read ORC footer for {path}: {error}"
                    ) from error

        num_rows = call_with_retry(
            read_num_rows,
            description=f"read ORC footer for {path}",
            match=DataContext.get_current().retried_io_errors,
        )
        if num_rows == 0:
            return None

        manifest = FileManifest.construct_manifest(
            paths=[path], sizes=[file_size], chunk_metadatas=[None]
        )
        batches = self._reader.read(manifest)
        try:
            sample = next(batches, None)
            if sample is None or sample.num_rows == 0 or sample.nbytes == 0:
                return None
            estimated_size = sample.nbytes * num_rows / sample.num_rows
            return estimated_size / file_size
        finally:
            if isinstance(batches, Generator):
                batches.close()
