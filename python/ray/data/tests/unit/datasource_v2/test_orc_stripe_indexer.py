"""Tests for optional ORC metadata, stripe planning, and physical reads."""

import importlib

import pyarrow as pa
import pyarrow.dataset as pds
import pyorc
import pytest
from pyarrow import orc
from pyarrow.fs import LocalFileSystem

from ray.data._internal.datasource_v2.formats.orc.orc_datasource_v2 import (
    OrcDatasourceV2,
)
from ray.data._internal.datasource_v2.formats.orc.orc_metadata import (
    orc_read_unit_id,
    pyorc_available,
    read_orc_metadata,
)
from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner
from ray.data._internal.datasource_v2.formats.orc.orc_stripe_indexer import (
    OrcStripeIndexer,
    OrcStripePartitioner,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import (
    FileChunk,
    FileManifest,
)
from ray.data._internal.datasource_v2.interfaces.file_partitioner import PartitionHints
from ray.data.context import DataContext
from ray.data.datasource.partitioning import Partitioning, PartitionStyle
from ray.data.expressions import col


def _write(path, rows=4096):
    table = pa.table(
        {"id": list(range(rows)), "label": [f"row-{i}" for i in range(rows)]}
    )
    orc.write_table(table, str(path), stripe_size=8192)
    return table


def _list(path, **kwargs):
    return list(
        OrcStripeIndexer(ignore_missing_paths=False).list_files(
            pa.array([str(path)]),
            filesystem=LocalFileSystem(),
            preserve_order=True,
            **kwargs,
        )
    )


def _without_pyorc(monkeypatch):
    original = importlib.import_module

    def load(name, package=None):
        if name == "pyorc":
            raise ModuleNotFoundError("No module named 'pyorc'", name="pyorc")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)


def test_metadata_does_not_iterate_data_rows(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    expected = _write(path)

    class MetadataOnlyReader(pyorc.Reader):
        def read(self, *args):
            raise AssertionError("metadata access must not decode rows")

        def __next__(self):
            raise AssertionError("metadata access must not decode rows")

    monkeypatch.setattr(pyorc, "Reader", MetadataOnlyReader)
    metadata = read_orc_metadata(str(path), None, statistic_columns={"id", "missing"})
    assert metadata is not None
    assert metadata.num_rows == len(expected)
    assert len(metadata.stripes) > 1
    assert sum(s.num_rows for s in metadata.stripes) == len(expected)
    assert metadata.stripes[0].row_offset == 0
    assert metadata.stripes[-1].row_offset + metadata.stripes[-1].num_rows == len(
        expected
    )
    assert sum(s.size_bytes for s in metadata.stripes) < path.stat().st_size
    assert metadata.statistics["id"]["minimum"] == 0
    assert metadata.statistics["id"]["maximum"] == len(expected) - 1
    assert "missing" not in metadata.statistics


def test_indexer_emits_exact_stripe_units_and_honors_exclusions(tmp_path):
    path = tmp_path / "data.orc"
    table = _write(path)
    manifests = _list(path)
    assert len(manifests) == orc.ORCFile(str(path)).nstripes
    chunks = [FileChunk.from_metadata(m.file_chunk_metadatas[0]) for m in manifests]
    assert sum(c.num_rows for c in chunks) == len(table)
    assert [c.unit_ids for c in chunks] == [(i,) for i in range(len(chunks))]
    selected = _list(path, excluded_read_unit_ids={orc_read_unit_id(str(path), 0)})
    assert len(selected) == len(manifests) - 1
    assert all(m.file_chunk_metadatas[0]["unit_ids"][0] != 0 for m in selected)
    assert _list(path, excluded_read_unit_ids={str(path)}) == []


def test_limit_only_counts_unfiltered_physical_rows(tmp_path):
    path = tmp_path / "data.orc"
    _write(path)
    manifests = _list(path, limit=1)
    assert len(manifests) == 1
    filtered = _list(path, limit=1, predicate=col("id") == 4095)
    assert len(filtered) == orc.ORCFile(str(path)).nstripes
    assert not any(m.file_chunk_metadatas[0]["fully_matched"] for m in filtered)


def test_overlapping_paths_do_not_duplicate_stripes(tmp_path):
    path = tmp_path / "data.orc"
    expected = _write(path)
    manifests = list(
        OrcStripeIndexer(ignore_missing_paths=False).list_files(
            pa.array([str(tmp_path), str(path)]),
            filesystem=LocalFileSystem(),
            preserve_order=True,
        )
    )
    assert len(manifests) == orc.ORCFile(str(path)).nstripes
    assert sum(m.file_chunk_metadatas[0]["num_rows"] for m in manifests) == len(
        expected
    )


def test_optional_backend_falls_back_to_whole_files(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    expected = _write(path)
    _without_pyorc(monkeypatch)
    assert not pyorc_available()
    assert read_orc_metadata(str(path), None) is None
    manifest = FileManifest.concat(_list(path))
    assert len(manifest) == 1
    assert manifest.file_chunk_metadatas.tolist() == [None]
    actual = pa.concat_tables(
        list(OrcScanner(schema=expected.schema).create_reader().read(manifest))
    )
    assert actual.equals(expected)


def test_corrupt_metadata_fails_with_path(tmp_path):
    path = tmp_path / "broken.orc"
    path.write_bytes(b"not an ORC file")
    with pytest.raises(pa.ArrowInvalid, match="broken.orc"):
        read_orc_metadata(str(path), None)


def test_empty_orc_has_no_read_units(tmp_path):
    path = tmp_path / "empty.orc"
    orc.write_table(pa.table({"id": pa.array([], type=pa.int64())}), str(path))
    metadata = read_orc_metadata(str(path), None)
    assert metadata is not None
    assert metadata.num_rows == 0
    assert _list(path) == []


@pytest.mark.parametrize("preserve_order", [False, True])
def test_chunked_reader_only_decodes_assigned_stripes(
    tmp_path, monkeypatch, preserve_order
):
    monkeypatch.setattr(
        DataContext.get_current().execution_options, "preserve_order", preserve_order
    )
    path = tmp_path / "data.orc"
    expected = _write(path)
    manifests = _list(path)
    selected = FileManifest.concat([manifests[1], manifests[-1]])
    decoded = []
    original = orc.ORCFile.read_stripe

    def read_stripe(self, index, columns=None):
        decoded.append(index)
        return original(self, index, columns)

    monkeypatch.setattr(orc.ORCFile, "read_stripe", read_stripe)
    reader = OrcScanner(schema=expected.schema, batch_size=128).create_reader()
    actual = pa.concat_tables(list(reader.read(selected)))
    metadata = read_orc_metadata(str(path), None)
    assert metadata is not None
    indices = [1, len(manifests) - 1]
    wanted = pa.concat_tables(
        expected.slice(metadata.stripes[i].row_offset, metadata.stripes[i].num_rows)
        for i in indices
    )
    if preserve_order:
        assert actual.equals(wanted)
    else:
        assert actual.sort_by("id").equals(wanted.sort_by("id"))
    assert sorted(decoded) == indices
    fragments = reader._get_fragments_to_read(
        pds.dataset(str(path), format="orc"), selected
    )
    assert [f.unit_start_row for f in fragments] == [
        metadata.stripes[i].row_offset for i in indices
    ]


def test_chunked_projection_reads_filter_columns(tmp_path):
    path = tmp_path / "data.orc"
    _write(path)
    manifest = FileManifest.concat(_list(path))
    scanner, residual = OrcScanner(
        schema=pa.schema([("id", pa.int64()), ("label", pa.string())]),
        batch_size=128,
    ).push_filters(col("id") >= 4090)
    assert residual is None
    actual = pa.concat_tables(
        list(
            scanner.prune_columns(["label"])
            .push_limit(3)
            .create_reader()
            .read(manifest)
        )
    )
    assert actual.to_pylist() == [{"label": f"row-{i}"} for i in range(4090, 4093)]


def test_chunk_retry_does_not_duplicate_rows(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    table = _write(path)
    manifest = _list(path)[0]
    reader = OrcScanner(schema=table.schema, batch_size=128).create_reader()
    original = reader._iter_stripe_tables
    failed = False

    def flaky(fragment, index, kwargs):
        nonlocal failed
        batches = original(fragment, index, kwargs)
        try:
            for i, batch in enumerate(batches):
                if i == 1 and not failed:
                    failed = True
                    raise OSError("ORC stripe transient probe")
                yield batch
        finally:
            batches.close()

    monkeypatch.setattr(reader, "_iter_stripe_tables", flaky)
    monkeypatch.setattr(
        DataContext.get_current(), "retried_io_errors", ["ORC stripe transient probe"]
    )
    monkeypatch.setattr("ray.data._internal.util.time.sleep", lambda _: None)
    actual = pa.concat_tables(list(reader.read(manifest)))
    assert failed
    assert actual.equals(table.slice(0, manifest.file_chunk_metadatas[0]["num_rows"]))


@pytest.mark.parametrize("preserve_order", [False, True])
def test_partitioner_preserves_all_units_and_creates_parallel_splits(
    tmp_path, monkeypatch, preserve_order
):
    monkeypatch.setattr(
        DataContext.get_current().execution_options, "preserve_order", preserve_order
    )
    path = tmp_path / "data.orc"
    table = _write(path)
    datasource = OrcDatasourceV2([str(path)])
    partitioner = datasource.get_file_partitioner(
        hints=PartitionHints(min_bucket_size=0, max_bucket_size=1 << 20, num_buckets=2)
    )
    assert isinstance(partitioner, OrcStripePartitioner)
    for manifest in _list(path):
        partitioner.add_input(manifest)
    partitioner.finalize()
    splits = []
    while partitioner.has_partition():
        splits.append(partitioner.next_partition())
    assert len(splits) >= 2
    actual = pa.concat_tables(
        batch
        for split in splits
        for batch in OrcScanner(schema=table.schema).create_reader().read(split)
    )
    if preserve_order:
        assert actual.equals(table)
    else:
        assert actual.sort_by("id").equals(table)


def test_small_file_coalesces_to_streamed_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    expected = _write(path)
    partitioner = OrcStripePartitioner(
        PartitionHints(
            min_bucket_size=1 << 20, max_bucket_size=128 << 20, num_buckets=200
        )
    )
    for manifest in _list(path):
        partitioner.add_input(manifest)
    partitioner.finalize()
    split = partitioner.next_partition()
    assert not partitioner.has_partition()

    def forbidden(*args, **kwargs):
        raise AssertionError("a coalesced whole file should use the batch scanner")

    monkeypatch.setattr(orc.ORCFile, "read_stripe", forbidden)
    actual = pa.concat_tables(
        list(OrcScanner(schema=expected.schema).create_reader().read(split))
    )
    assert actual.equals(expected)


def test_missing_backend_on_read_worker_does_not_read_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    table = _write(path)
    manifest = _list(path)[0]
    _without_pyorc(monkeypatch)
    with pytest.raises(ModuleNotFoundError, match="every read worker"):
        list(OrcScanner(schema=table.schema).create_reader().read(manifest))


@pytest.mark.parametrize("conflict", [False, True])
def test_chunked_partition_validation_precedes_projection(tmp_path, conflict):
    folder = tmp_path / "year=2024"
    folder.mkdir()
    path = folder / "data.orc"
    years = ["wrong"] * 4096 if conflict else [None, "2024"] * 2048
    table = pa.table(
        {"id": list(range(4096)), "year": pa.array(years, type=pa.string())}
    )
    orc.write_table(table, str(path), stripe_size=8192)
    assert orc.ORCFile(str(path)).nstripes > 1
    manifest = _list(path)[0]
    reader = (
        OrcScanner(
            schema=table.schema,
            batch_size=64,
            partitioning=Partitioning(PartitionStyle.HIVE, base_dir=str(tmp_path)),
        )
        .prune_columns(["id"])
        .create_reader()
    )
    if conflict:
        with pytest.raises(ValueError, match="Partition column year"):
            list(reader.read(manifest))
    else:
        actual = pa.concat_tables(list(reader.read(manifest)))
        assert actual.to_pylist() == [
            {"id": i} for i in range(manifest.file_chunk_metadatas[0]["num_rows"])
        ]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
