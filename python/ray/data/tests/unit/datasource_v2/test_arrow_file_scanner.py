"""Unit tests for :meth:`FileScanner.prune_input_split`."""
import pyarrow as pa
import pytest

from ray.data._internal.datasource_v2.formats.parquet.parquet_scanner import (
    ParquetScanner,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import (
    FILE_CHUNK_METADATA_COLUMN_NAME,
    FILE_SIZE_COLUMN_NAME,
    PATH_COLUMN_NAME,
    FileManifest,
)
from ray.data.datasource.partitioning import Partitioning, PartitionStyle
from ray.data.expressions import col


def test_prune_input_split_matching_no_file():
    """Pruning away every file is an ordinary outcome, not an error."""
    manifest = FileManifest(
        pa.table(
            {
                PATH_COLUMN_NAME: ["/root/year=2020/data.parquet"],
                FILE_SIZE_COLUMN_NAME: [1],
                FILE_CHUNK_METADATA_COLUMN_NAME: [None],
            }
        )
    )
    scanner = ParquetScanner(
        schema=pa.schema([("x", pa.int64())]),
        partitioning=Partitioning(PartitionStyle.HIVE, base_dir="/root"),
        partition_predicate=col("year") == "2029",
    )

    assert len(scanner.prune_input_split(manifest)) == 0


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("physical_type", [pa.int64(), pa.string()])
def test_shared_schema_preserves_format_specific_synthesized_types(
    projected, physical_type
):
    from ray.data._internal.datasource_v2.common.synthesized_columns import PathColumn
    from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner

    physical = pa.field("path", physical_type, metadata={b"source": b"file"})
    schema = pa.schema([("id", pa.int64()), physical])
    columns = ("path", "id") if projected else None
    parquet = ParquetScanner(
        schema=schema, columns=columns, synthesized_columns=(PathColumn(),)
    )
    orc = OrcScanner(
        schema=schema, columns=columns, synthesized_columns=(PathColumn(),)
    )
    assert parquet.read_schema().field("path") == physical
    assert parquet.read_schema().field("path").metadata == physical.metadata
    field = orc.read_schema().field("path")
    assert field.type == pa.string()
    if projected or physical_type != pa.string():
        assert field.metadata is None
    else:
        assert field.metadata == physical.metadata
    expected_names = ["path", "id"] if projected else ["id", "path"]
    assert parquet.read_schema().names == expected_names
    assert orc.read_schema().names == expected_names


def test_shared_schema_preserves_missing_projected_field_policy():
    from ray.data._internal.datasource_v2.common.synthesized_columns import PathColumn
    from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner

    kwargs = dict(
        schema=pa.schema([("id", pa.int64())]),
        columns=("path",),
        synthesized_columns=(PathColumn(),),
    )
    assert OrcScanner(**kwargs).read_schema() == pa.schema([("path", pa.string())])
    with pytest.raises(AssertionError, match="Column path not found"):
        ParquetScanner(**kwargs).read_schema()


@pytest.mark.parametrize("columns", [None, (), ("id",)])
def test_shared_schema_handles_projection_of_synthesized_fields(columns):
    from ray.data._internal.datasource_v2.common.synthesized_columns import PathColumn
    from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner

    for scanner_type in (ParquetScanner, OrcScanner):
        scanner = scanner_type(
            schema=pa.schema([("id", pa.int64())]),
            columns=columns,
            synthesized_columns=(PathColumn(),),
        )
        expected = ["id", "path"] if columns is None else list(columns)
        assert scanner.read_schema().names == expected


def test_shared_schema_retains_parquet_tensor_check(monkeypatch):
    from ray.data._internal.datasource_v2.formats.parquet import parquet_scanner

    checked = []
    monkeypatch.setattr(parquet_scanner, "check_for_legacy_tensor_type", checked.append)
    schema = pa.schema([("id", pa.int64())])
    assert ParquetScanner(schema=schema).read_schema() == schema
    assert checked == [schema]


def test_shared_schema_propagates_parquet_tensor_check_errors(monkeypatch):
    from ray.data._internal.datasource_v2.formats.parquet import parquet_scanner

    def reject_legacy_schema(_):
        raise RuntimeError("legacy tensor type")

    monkeypatch.setattr(
        parquet_scanner, "check_for_legacy_tensor_type", reject_legacy_schema
    )
    with pytest.raises(RuntimeError, match="legacy tensor type"):
        ParquetScanner(schema=pa.schema([("id", pa.int64())])).read_schema()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
