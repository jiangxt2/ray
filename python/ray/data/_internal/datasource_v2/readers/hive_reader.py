"""Sequential, bounded-batch HiveServer2 result reader."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Optional, Tuple

import pyarrow as pa
from typing_extensions import override

from ray.data._internal.datasource.hive_hs2_client import (
    HiveClientCompatibilityError,
    HiveConnectionOptions,
    HiveResponseTooLargeError,
    connect_hiveserver2,
)
from ray.data._internal.datasource_v2.hive_types import (
    HiveTypeCompatibilityError,
    schema_from_description,
    schemas_match,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.readers.base_reader import Reader

DEFAULT_HIVE_TARGET_BATCH_BYTES = 1024 * 1024
DEFAULT_HIVE_TASK_MEMORY_BYTES = 128 * 1024 * 1024
DEFAULT_HIVE_INITIAL_BATCH_ROWS = 1
MAX_HIVE_BATCH_ROWS = 512
_HIVE_BATCH_SAFETY_FACTOR = 3.0
_HIVE_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?$"
)
_EPOCH = datetime.datetime(1970, 1, 1)
_NANOSECONDS_PER_SECOND = 1_000_000_000
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _parse_hive_timestamp_ns(value: str) -> int:
    """Parse a Hive timestamp string without losing sub-microsecond digits."""
    match = _HIVE_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise HiveTypeCompatibilityError(
            "Hive timestamp value does not match the supported format"
        )

    try:
        timestamp = datetime.datetime.fromisoformat(match.group(1))
    except ValueError:
        raise HiveTypeCompatibilityError(
            "Hive timestamp value is outside the supported calendar range"
        ) from None

    delta = timestamp - _EPOCH
    whole_seconds = delta.days * 86_400 + delta.seconds
    fractional_ns = int((match.group(2) or "").ljust(9, "0"))
    timestamp_ns = whole_seconds * _NANOSECONDS_PER_SECOND + fractional_ns
    if not _INT64_MIN <= timestamp_ns <= _INT64_MAX:
        raise HiveTypeCompatibilityError(
            "Hive timestamp value is outside Arrow's nanosecond range"
        )
    return timestamp_ns


class _AdaptiveBatchSizer:
    """Choose the next request row count from recent Arrow and wire sizes."""

    def __init__(self, target_bytes: int, column_count: int):
        self._target_bytes = max(1, target_bytes)
        # `CBatch.pop_many()` creates one Python tuple/reference per cell.
        # Account for that row-width overhead even when Arrow values are null
        # or zero-width in the sampled batch.
        self._row_width_floor = max(1, column_count) * 64
        self._estimated_row_bytes: Optional[float] = None

    def observe(self, table: pa.Table, wire_bytes: int) -> int:
        if table.num_rows > 0:
            observed = max(
                self._row_width_floor,
                table.nbytes / table.num_rows,
                wire_bytes / table.num_rows,
            )
            if self._estimated_row_bytes is None:
                self._estimated_row_bytes = observed
            else:
                # Follow increases immediately; decay slowly after a large batch
                # so the next request remains conservative under skew.
                self._estimated_row_bytes = max(
                    observed, self._estimated_row_bytes * 0.8 + observed * 0.2
                )

        row_bytes = max(self._estimated_row_bytes or 1.0, 1.0)
        rows = int(self._target_bytes / (row_bytes * _HIVE_BATCH_SAFETY_FACTOR))
        return min(max(rows, 1), MAX_HIVE_BATCH_ROWS)


def _rows_to_arrow(rows: list[Tuple[Any, ...]], schema: pa.Schema) -> pa.Table:
    if not rows:
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in schema], schema=schema
        )

    column_count = len(schema)
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != column_count:
            raise HiveTypeCompatibilityError(
                f"Hive result row {row_index} does not match the declared schema"
            )

    arrays = []
    for column_index, arrow_field in enumerate(schema):
        values = [row[column_index] for row in rows]
        if pa.types.is_date(arrow_field.type):
            values = [
                datetime.date.fromisoformat(value)
                if isinstance(value, str) and value
                else value
                for value in values
            ]
        elif pa.types.is_timestamp(arrow_field.type):
            if any(
                value is not None and not isinstance(value, str) for value in values
            ):
                raise HiveTypeCompatibilityError(
                    "Hive timestamp values must retain their original string form"
                )
            try:
                timestamp_values = [
                    _parse_hive_timestamp_ns(value) if value is not None else None
                    for value in values
                ]
                arrays.append(
                    pa.array(timestamp_values, type=pa.int64()).cast(arrow_field.type)
                )
            except HiveTypeCompatibilityError:
                raise
            except Exception:
                raise HiveTypeCompatibilityError(
                    "Hive result values do not match the declared type for column "
                    f"{arrow_field.name!r}"
                ) from None
            continue
        elif pa.types.is_decimal(arrow_field.type):
            try:
                values = [
                    Decimal(value) if isinstance(value, str) and value else value
                    for value in values
                ]
            except InvalidOperation:
                raise HiveTypeCompatibilityError(
                    "Hive result values do not match the declared type for column "
                    f"{arrow_field.name!r}"
                ) from None
        try:
            arrays.append(pa.array(values, type=arrow_field.type))
        except Exception:
            raise HiveTypeCompatibilityError(
                "Hive result values do not match the declared type for column "
                f"{arrow_field.name!r}"
            ) from None
    return pa.Table.from_arrays(arrays, schema=schema)


def _schema_summary(schema: pa.Schema) -> str:
    fields = [f"{field.name[:80]!r}: {field.type}" for field in list(schema)[:8]]
    if len(schema) > 8:
        fields.append(f"... {len(schema) - 8} more")
    return f"{len(schema)} column(s) [{', '.join(fields)}]"


@dataclass(frozen=True)
class HiveReader(Reader[FileManifest]):
    connection_options: HiveConnectionOptions = field(repr=False)
    query: str = field(repr=False)
    expected_query_schema: pa.Schema = field(repr=False)
    output_schema: pa.Schema = field(repr=False)
    local_output_columns: Optional[Tuple[str, ...]] = None
    target_batch_bytes: int = DEFAULT_HIVE_TARGET_BATCH_BYTES

    @override
    def read(self, input_split: FileManifest) -> Iterator[pa.Table]:
        if len(input_split) == 0:
            return
        if len(input_split) != 1:
            raise ValueError("strict HiveServer2 reads require one manifest row")

        connection = None
        cursor = None
        operation = None
        message_transport = None
        completed = False
        response_too_large = False
        try:
            connection, message_transport = connect_hiveserver2(
                host=self.connection_options.host,
                port=self.connection_options.port,
                auth_mechanism=self.connection_options.auth_mechanism,
                user=self.connection_options.user,
                password=self.connection_options.password,
                kerberos_service_name=self.connection_options.kerberos_service_name,
                use_ssl=self.connection_options.use_ssl,
                ca_cert=self.connection_options.ca_cert,
                timeout=self.connection_options.timeout,
            )
            cursor = connection.cursor(
                user=self.connection_options.user, convert_types=False
            )
            cursor.execute(self.query)

            description = cursor.description
            actual_schema = schema_from_description(description)
            if not schemas_match(actual_schema, self.expected_query_schema):
                raise HiveTypeCompatibilityError(
                    "Hive result metadata changed after planning; refusing to emit "
                    "blocks with a mismatched schema. Planned: "
                    f"{_schema_summary(self.expected_query_schema)}; query: "
                    f"{_schema_summary(actual_schema)}"
                )

            operation = cursor._last_operation
            if operation is None or not cursor.has_result_set:
                raise HiveTypeCompatibilityError(
                    "Hive query did not return a result set"
                )
            has_timestamps = any(
                pa.types.is_timestamp(field.type) for field in actual_schema
            )
            if has_timestamps and not getattr(operation, "is_columnar", False):
                raise HiveTypeCompatibilityError(
                    "Hive timestamp reads require columnar results to preserve precision"
                )

            sizer = _AdaptiveBatchSizer(
                self.target_batch_bytes, column_count=len(actual_schema)
            )
            requested_rows = DEFAULT_HIVE_INITIAL_BATCH_ROWS
            empty_batches = 0
            while True:
                # `Operation.fetch` issues exactly one FetchResults RPC. The
                # public DB-API `fetchmany()` may combine multiple RPC responses
                # into one list if the server returns short batches.
                cursor.arraysize = requested_rows
                batch = operation.fetch(
                    description,
                    max_rows=requested_rows,
                    convert_types=cursor.convert_types,
                    convert_strings_to_unicode=cursor.convert_strings_to_unicode,
                )
                if batch is None:
                    break
                row_count = len(batch)
                expect_more_rows = bool(getattr(batch, "expect_more_rows", False))
                if row_count == 0:
                    if not expect_more_rows:
                        break
                    empty_batches += 1
                    if empty_batches > 100:
                        raise RuntimeError(
                            "HiveServer2 returned too many empty result batches"
                        )
                    continue

                # HS2 marks hasMoreRows optional. Impyla maps an omitted value
                # to False, so keep fetching after every non-empty batch and
                # use an empty batch with False as the end-of-stream signal.
                empty_batches = 0
                rows = batch.pop_many(row_count)
                table = _rows_to_arrow(rows, actual_schema)
                wire_bytes = message_transport.last_message_bytes
                requested_rows = sizer.observe(table, wire_bytes)
                if self.local_output_columns is not None:
                    table = table.select(list(self.local_output_columns))
                if not schemas_match(table.schema, self.output_schema):
                    raise HiveTypeCompatibilityError(
                        "Hive Arrow conversion did not match the planned output schema"
                    )
                del rows, batch
                yield table
            completed = True
        except GeneratorExit:
            raise
        except HiveResponseTooLargeError:
            response_too_large = True
            if message_transport is not None:
                try:
                    # An over-limit message leaves unread bytes on the Thrift
                    # stream; close it instead of sending another RPC on it.
                    message_transport.close()
                except Exception:
                    pass
            raise
        except HiveTypeCompatibilityError:
            raise
        except HiveClientCompatibilityError:
            raise
        except Exception as exc:
            # Do not include SQL, row values, credentials, or a driver's raw
            # connection details in the datasource's public error message.
            raise RuntimeError(
                f"HiveServer2 read failed ({type(exc).__name__}); "
                "the query was not retried"
            ) from None
        finally:
            if response_too_large:
                # The stream is desynchronized; do not send Cancel/Close RPCs
                # on it. Closing the transport tears down the HS2 session.
                if cursor is not None and hasattr(cursor, "_closed"):
                    cursor._closed = True
            else:
                if not completed and cursor is not None:
                    try:
                        cursor.cancel_operation()
                    except Exception:
                        pass
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
