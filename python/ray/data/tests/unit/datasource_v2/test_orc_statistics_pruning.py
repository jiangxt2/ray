"""Correctness and physical-read tests for conservative ORC pruning."""

import pyarrow as pa
import pyarrow.dataset as pds
import pyorc
import pytest
from pyarrow import orc
from pyarrow.fs import LocalFileSystem

from ray.data._internal.datasource_v2.formats.orc.orc_datasource_v2 import (
    OrcDatasourceV2,
)
from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner
from ray.data._internal.datasource_v2.formats.orc.orc_statistics import (
    OrcStatisticsIndexer,
    statistics_can_match,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data.datasource.partitioning import Partitioning, PartitionStyle
from ray.data.expressions import col, lit


def _stats(low=10, high=20, values=3, kind=pyorc.TypeKind.LONG):
    return {
        "id": {
            "minimum": low,
            "maximum": high,
            "number_of_values": values,
            "kind": kind,
        }
    }


@pytest.mark.parametrize(
    "predicate,possible",
    [
        (col("id") == 9, False),
        (col("id") == 10, True),
        (col("id") == 20, True),
        (col("id") == 21, False),
        (col("id") < 10, False),
        (col("id") <= 10, True),
        (col("id") > 20, False),
        (col("id") >= 20, True),
        (lit(9) >= col("id"), False),
        ((col("id") < 10) & (col("missing") > 0), False),
        ((col("id") < 10) | (col("missing") > 0), True),
        ((col("id") < 10) | (col("id") > 20), False),
        (~(col("id") > 20), True),
        (col("id").is_null(), True),
        (col("id") == lit(None), True),
        (col("id") == lit(True), True),
        (col("id") == 1.5, True),
        (col("id") == (1 << 100), True),
    ],
)
def test_statistics_comparison_boundaries(predicate, possible):
    assert statistics_can_match(predicate, _stats()) is possible


def test_all_null_and_constant_statistics():
    assert not statistics_can_match(col("id") == 10, _stats(None, None, 0))
    assert not statistics_can_match(col("id") != 10, _stats(10, 10))
    assert statistics_can_match(col("id") != 10, _stats(10, 20))


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"id": {}},
        _stats(None, None),
        _stats(20, 10),
        _stats(float("nan"), 20, kind=pyorc.TypeKind.DOUBLE),
        _stats(10, 20, kind=pyorc.TypeKind.TIMESTAMP),
        _stats("10", "20"),
    ],
)
def test_unknown_or_untrusted_statistics_keep_units(stats):
    assert statistics_can_match(col("id") == 100, stats)


def test_string_bounds_and_unknown_truncated_bounds():
    stats = {
        "id": {
            "kind": pyorc.TypeKind.STRING,
            "minimum": "a",
            "maximum": "z",
            "number_of_values": 2,
        }
    }
    assert not statistics_can_match(col("id") == "λ", stats)
    assert statistics_can_match(col("id") == "a", stats)
    stats["id"].pop("maximum")
    stats["id"]["upper_bound"] = "z"
    assert statistics_can_match(col("id") == "λ", stats)


def test_integer_promotion_rounding_is_conservative():
    value = (1 << 53) + 1
    # Native integer statistics cannot reject a value that matches after
    # conversion to the unified double schema.
    assert statistics_can_match(col("id") == (1 << 53), _stats(value, value))
    assert statistics_can_match(
        col("id") == (1 << 24), _stats((1 << 24) + 1, (1 << 24) + 1)
    )


def _manifests(path, predicate, **kwargs):
    return list(
        OrcStatisticsIndexer(
            ignore_missing_paths=False, enable_pruning=True
        ).list_files(
            pa.array([str(path)]),
            filesystem=LocalFileSystem(),
            predicate=predicate,
            preserve_order=True,
            **kwargs,
        )
    )


def test_pruning_reduces_actual_stripe_decodes(tmp_path, monkeypatch):
    path = tmp_path / "data.orc"
    expected = pa.table(
        {"id": list(range(8192)), "label": [f"row-{i}" for i in range(8192)]}
    )
    orc.write_table(expected, str(path), stripe_size=8192)
    predicate = col("id") >= 8100
    manifests = _manifests(path, predicate, limit=1)
    assert manifests
    selected = [
        int(i)
        for manifest in manifests
        for i in manifest.file_chunk_metadatas[0]["unit_ids"]
    ]
    assert len(selected) < orc.ORCFile(str(path)).nstripes
    assert not any(m.file_chunk_metadatas[0]["fully_matched"] for m in manifests)
    decoded = []
    original = orc.ORCFile.read_stripe

    def read(self, index, columns=None):
        decoded.append(index)
        return original(self, index, columns)

    monkeypatch.setattr(orc.ORCFile, "read_stripe", read)
    scanner, residual = OrcScanner(schema=expected.schema).push_filters(predicate)
    assert residual is None
    actual = pa.concat_tables(
        list(scanner.create_reader().read(FileManifest.concat(manifests)))
    )
    assert actual.sort_by("id").equals(expected.slice(8100))
    assert sorted(decoded) == selected
    assert _manifests(path, col("id") < 0) == []


def test_missing_columns_and_float_nan_keep_files(tmp_path):
    path = tmp_path / "floats.orc"
    orc.write_table(pa.table({"id": [1.0, float("nan"), None]}), str(path))
    assert _manifests(path, col("id") > 100)
    assert _manifests(path, col("missing") == 1)


def test_partitioned_inputs_preserve_validation_before_filter(tmp_path):
    folder = tmp_path / "year=2024"
    folder.mkdir()
    path = folder / "data.orc"
    orc.write_table(pa.table({"id": [100], "year": ["wrong"]}), str(path))
    datasource = OrcDatasourceV2(
        [str(path)],
        partitioning=Partitioning(PartitionStyle.HIVE, base_dir=str(tmp_path)),
    )
    indexer = datasource._get_file_indexer()
    assert isinstance(indexer, OrcStatisticsIndexer)
    manifests = list(
        indexer.list_files(
            pa.array(datasource.paths),
            filesystem=datasource.filesystem,
            predicate=col("id") < 0,
        )
    )
    assert manifests
    scanner = datasource.create_scanner(
        pa.schema([("id", pa.int64()), ("year", pa.string())])
    )
    _, residual = scanner.push_filters(col("id") < 0)
    assert residual is not None
    with pytest.raises(ValueError, match="Partition column year"):
        list(scanner.create_reader().read(FileManifest.concat(manifests)))


def test_integer_to_double_alignment_error_is_not_hidden(tmp_path):
    int_path, float_path = tmp_path / "int.orc", tmp_path / "float.orc"
    orc.write_table(pa.table({"id": [(1 << 53) + 1]}), str(int_path))
    orc.write_table(pa.table({"id": [0.0]}), str(float_path))
    predicate = col("id") == (1 << 53)
    manifests = _manifests(int_path, predicate)
    assert manifests
    schema = pa.schema([("id", pa.float64())])
    # The baseline uses Arrow's safe cast, which rejects lossy alignment.
    with pytest.raises(pa.ArrowInvalid, match="not in range"):
        fragment = next(pds.dataset(str(int_path), format="orc").get_fragments())
        fragment.scanner(schema=schema, filter=predicate.to_pyarrow()).to_table()
    scanner, _ = OrcScanner(schema=schema).push_filters(predicate)
    with pytest.raises(pa.ArrowInvalid, match="not in range"):
        list(scanner.create_reader().read(FileManifest.concat(manifests)))


def test_long_utf8_string_statistics_never_prune_matching_rows(tmp_path):
    path = tmp_path / "strings.orc"
    value = "λ" + "x" * 2048
    expected = pa.table({"id": [value, value + "z"]})
    orc.write_table(expected, str(path))
    predicate = col("id") == value + "z"
    manifests = _manifests(path, predicate)
    assert manifests
    scanner, _ = OrcScanner(schema=expected.schema).push_filters(predicate)
    actual = pa.concat_tables(
        list(scanner.create_reader().read(FileManifest.concat(manifests)))
    )
    assert actual.to_pylist() == [{"id": value + "z"}]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
