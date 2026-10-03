import os

import numpy as np
import pyarrow as pa
import pyarrow.orc as orc
import pytest

from ray.data._internal.datasource_v2.formats.orc.orc_size_estimator import (
    OrcInMemorySizeEstimator,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest


def _write(path, table, compression="zlib"):
    orc.write_table(table, str(path), compression=compression)


def _manifest(*paths):
    return FileManifest.construct_manifest(
        paths=list(map(str, paths)),
        sizes=[os.path.getsize(path) for path in paths],
        chunk_metadatas=[None] * len(paths),
    )


@pytest.mark.parametrize("compression", ["uncompressed", "zlib"])
@pytest.mark.parametrize("width", [8, 1024])
def test_estimates_decoded_size_for_multiple_batches(tmp_path, compression, width):
    path = tmp_path / "data.orc"
    table = pa.table({"id": range(5000), "payload": ["x" * width] * 5000})
    _write(path, table, compression)
    estimate = OrcInMemorySizeEstimator().estimate_in_memory_sizes(_manifest(path))
    assert estimate[0] == pytest.approx(table.nbytes, rel=0.01)


def test_estimator_reads_one_batch_and_closes_iterator(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    table = pa.table({"id": range(5000)})
    _write(path, table)
    estimator = OrcInMemorySizeEstimator()
    closed = []

    def read(_):
        try:
            yield table.slice(0, 10)
            pytest.fail("The estimator requested another batch")
        finally:
            closed.append(True)

    monkeypatch.setattr(estimator._reader, "read", read)
    assert estimator.estimate_in_memory_sizes(_manifest(path))[0] == pytest.approx(
        table.nbytes
    )
    assert closed == [True]


def test_estimator_reuses_ratio_without_more_io(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    table = pa.table({"id": range(5000)})
    _write(path, table)
    manifest = _manifest(path)
    estimator = OrcInMemorySizeEstimator()
    first = estimator.estimate_in_memory_sizes(manifest)
    monkeypatch.setattr(
        estimator, "_estimate_encoding_ratio", lambda *_: pytest.fail("resampled")
    )
    assert np.array_equal(estimator.estimate_in_memory_sizes(manifest), first)


def test_empty_file_does_not_lock_in_zero_ratio(tmp_path):
    empty = tmp_path / "empty.orc"
    populated = tmp_path / "data.orc"
    _write(empty, pa.table({"id": pa.array([], type=pa.int64())}))
    table = pa.table({"id": range(2000)})
    _write(populated, table)
    estimator = OrcInMemorySizeEstimator()
    manifest = _manifest(empty)
    assert np.array_equal(
        estimator.estimate_in_memory_sizes(manifest), manifest.file_sizes
    )
    assert estimator._encoding_ratio is None
    assert estimator.estimate_in_memory_sizes(_manifest(populated))[0] == pytest.approx(
        table.nbytes
    )


def test_estimator_skips_empty_first_file(tmp_path):
    empty = tmp_path / "empty.orc"
    populated = tmp_path / "data.orc"
    _write(empty, pa.table({"id": pa.array([], type=pa.int64())}))
    table = pa.table({"id": range(2000)})
    _write(populated, table)
    estimate = OrcInMemorySizeEstimator().estimate_in_memory_sizes(
        _manifest(empty, populated)
    )
    assert estimate[1] == pytest.approx(table.nbytes)


def test_estimator_empty_manifest():
    result = OrcInMemorySizeEstimator().estimate_in_memory_sizes(
        FileManifest.construct_manifest(paths=[], sizes=[], chunk_metadatas=[])
    )
    assert len(result) == 0


def test_estimator_rejects_corrupt_footer(tmp_path):
    path = tmp_path / "broken.orc"
    path.write_bytes(b"not an ORC file")
    with pytest.raises((pa.ArrowInvalid, OSError)) as exc:
        OrcInMemorySizeEstimator().estimate_in_memory_sizes(_manifest(path))
    assert str(path) in str(exc.value)


def test_estimator_improves_default_multi_batch_fallback(tmp_path):
    from ray.data._internal.datasource_v2.common.file_reader import FileFormat
    from ray.data._internal.datasource_v2.common.size_estimators import (
        SamplingInMemorySizeEstimator,
    )
    from ray.data._internal.datasource_v2.formats.orc.orc_file_reader import (
        OrcFileReader,
    )

    path = tmp_path / "large.orc"
    table = pa.table({"id": range(140000), "payload": ["x" * 64] * 140000})
    _write(path, table)
    manifest = _manifest(path)
    old = SamplingInMemorySizeEstimator(OrcFileReader(format=FileFormat.ORC))
    assert old.estimate_in_memory_sizes(manifest)[0] == manifest.file_sizes[0]
    new = OrcInMemorySizeEstimator().estimate_in_memory_sizes(manifest)[0]
    assert new == pytest.approx(table.nbytes, rel=0.01)
    assert abs(new - table.nbytes) < abs(manifest.file_sizes[0] - table.nbytes)


def test_estimator_retries_footer_io(tmp_path, monkeypatch):
    from ray.data.context import DataContext

    path = tmp_path / "data.orc"
    table = pa.table({"id": range(2000)})
    _write(path, table)
    original = orc.ORCFile
    calls = []
    backoffs = []

    def open_orc(source):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("temporary footer failure")
        return original(source)

    monkeypatch.setattr(DataContext.get_current(), "retried_io_errors", ["temporary"])
    monkeypatch.setattr("ray._common.retry.time.sleep", backoffs.append)
    monkeypatch.setattr(orc, "ORCFile", open_orc)
    estimate = OrcInMemorySizeEstimator().estimate_in_memory_sizes(_manifest(path))
    assert estimate[0] == pytest.approx(table.nbytes)
    assert len(calls) == 2
    assert len(backoffs) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
