"""Optional, metadata-only PyORC access for ORC read units."""

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import pyarrow as pa
from pyarrow.fs import FileSystem, LocalFileSystem

from ray.data._internal.util import call_with_retry
from ray.data.context import DataContext


@dataclass(frozen=True)
class OrcStripeMetadata:
    index: int
    num_rows: int
    row_offset: int
    size_bytes: int
    statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class OrcFileMetadata:
    num_rows: int
    stripes: Tuple[OrcStripeMetadata, ...]
    statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def pyorc_available() -> bool:
    """Return whether the optional metadata backend is installed."""
    try:
        importlib.import_module("pyorc")
    except ModuleNotFoundError as error:
        if error.name == "pyorc":
            return False
        raise
    return True


def orc_read_unit_id(path: str, stripe_index: int) -> str:
    """Name a physical stripe independently of task grouping."""
    return f"{path}#stripe{stripe_index}"


def read_orc_metadata(
    path: str,
    filesystem: Optional[FileSystem],
    *,
    statistic_columns: Optional[Set[str]] = None,
) -> Optional[OrcFileMetadata]:
    """Read stripe layout and requested column statistics without decoding rows."""
    if not pyorc_available():
        return None
    backend = importlib.import_module("pyorc")
    reader_factory = backend.Reader
    column_factory = backend.Column
    orc_error: type[Exception] = backend.ORCError
    filesystem = filesystem or LocalFileSystem()

    def read() -> OrcFileMetadata:
        with filesystem.open_input_file(path) as source:
            try:
                reader = reader_factory(source)
                fields = getattr(reader.schema, "fields", {})
                column_ids = {
                    name: fields[name].column_id
                    for name in statistic_columns or ()
                    if name in fields
                }

                def statistics(stream) -> Dict[str, Dict[str, Any]]:
                    return {
                        name: dict(column_factory(stream, index).statistics)
                        for name, index in column_ids.items()
                    }

                stripes = tuple(
                    OrcStripeMetadata(
                        index=index,
                        num_rows=len(stripe),
                        row_offset=stripe.row_offset,
                        size_bytes=stripe.bytes_length,
                        statistics=statistics(stripe),
                    )
                    for index, stripe in enumerate(reader.iter_stripes())
                )
                return OrcFileMetadata(
                    num_rows=len(reader),
                    stripes=stripes,
                    statistics=statistics(reader),
                )
            except (RuntimeError, ValueError, orc_error) as error:
                raise pa.ArrowInvalid(
                    f"Failed to read ORC metadata for {path}: {error}"
                ) from error

    return call_with_retry(
        read,
        description=f"read ORC metadata for {path}",
        match=DataContext.get_current().retried_io_errors,
    )
