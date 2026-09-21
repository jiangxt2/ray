from dataclasses import dataclass
from decimal import Decimal

import pyarrow as pa
import pytest

from ray.data._internal.datasource.hive_hs2_client import (
    HiveClientCompatibilityError,
    HiveConnectionOptions,
    HiveResponseTooLargeError,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.readers import hive_reader
from ray.data._internal.datasource_v2.readers.hive_reader import HiveReader


@dataclass
class _FakeBatch:
    rows: list
    expect_more_rows: bool

    def __len__(self):
        return len(self.rows)

    def pop_many(self, size):
        rows, self.rows = self.rows[:size], self.rows[size:]
        return rows


class _FakeMessageTransport:
    def __init__(self):
        self.last_message_bytes = 64
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakeOperation:
    def __init__(self, batches, message_transport):
        self.batches = list(batches)
        self.message_transport = message_transport
        self.fetch_calls = []
        self.is_columnar = True

    def fetch(
        self,
        schema,
        max_rows,
        convert_types=True,
        convert_strings_to_unicode=True,
    ):
        self.fetch_calls.append(max_rows)
        self.message_transport.last_message_bytes = 64
        if not self.batches:
            return _FakeBatch([], False)
        return self.batches.pop(0)


class _FakeCursor:
    description = [("id", "INT", None, None, None, None, None)]
    convert_types = True
    convert_strings_to_unicode = True
    has_result_set = True

    def __init__(self, operation):
        self._last_operation = operation
        self.execute_calls = []
        self.cancel_calls = 0
        self.close_calls = 0
        self.arraysize = None
        self._closed = False

    def execute(self, query):
        self.execute_calls.append(query)

    def cancel_operation(self):
        self.cancel_calls += 1

    def close(self):
        self.close_calls += 1


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.close_calls = 0

    def cursor(self, user=None, convert_types=True):
        self._cursor.convert_types = convert_types
        return self._cursor

    def close(self):
        self.close_calls += 1


def _reader():
    return HiveReader(
        connection_options=HiveConnectionOptions(
            host="localhost", port=10000, auth_mechanism="NOSASL"
        ),
        query="SELECT `id` FROM `default`.`t`",
        expected_query_schema=pa.schema([("id", pa.int32())]),
        output_schema=pa.schema([("id", pa.int32())]),
        target_batch_bytes=1024,
    )


def _manifest():
    return FileManifest.construct_manifest(
        paths=["hive://planned-read"], sizes=[1], chunk_metadatas=[None]
    )


def test_hive_reader_uses_one_query_and_single_response_fetches(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation(
        [
            _FakeBatch([(1,)], False),
            _FakeBatch([(2,), (3,)], False),
        ],
        transport,
    )
    cursor = _FakeCursor(operation)
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )

    tables = list(_reader().read(_manifest()))

    assert [table.column("id").to_pylist() for table in tables] == [[1], [2, 3]]
    assert cursor.execute_calls == ["SELECT `id` FROM `default`.`t`"]
    assert operation.fetch_calls[0] == 1
    assert len(operation.fetch_calls) == 3
    assert cursor.convert_types is False
    assert cursor.close_calls == 1
    assert connection.close_calls == 1
    assert cursor.cancel_calls == 0


def test_hive_reader_propagates_oversized_response_and_cleans_up(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation([], transport)

    def fail_fetch(*args, **kwargs):
        raise HiveResponseTooLargeError("response exceeds cap")

    operation.fetch = fail_fetch
    cursor = _FakeCursor(operation)
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )

    with pytest.raises(HiveResponseTooLargeError):
        list(_reader().read(_manifest()))

    assert cursor.cancel_calls == 0
    assert cursor.close_calls == 0
    assert connection.close_calls == 0
    assert cursor._closed is True
    assert transport.close_calls == 1


def test_hive_reader_preserves_impyla_compatibility_error(monkeypatch):
    transport = _FakeMessageTransport()

    def fail_connect(**kwargs):
        raise HiveClientCompatibilityError("unsupported Impyla version")

    monkeypatch.setattr(hive_reader, "connect_hiveserver2", fail_connect)

    with pytest.raises(HiveClientCompatibilityError, match="unsupported Impyla"):
        list(_reader().read(_manifest()))

    assert transport.close_calls == 0


def test_hive_reader_rejects_result_schema_change_before_emitting(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation([_FakeBatch([(1,)], False)], transport)
    cursor = _FakeCursor(operation)
    cursor.description = [("other", "INT", None, None, None, None, None)]
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )

    with pytest.raises(ValueError, match="metadata changed"):
        list(_reader().read(_manifest()))

    assert operation.fetch_calls == []
    assert cursor.cancel_calls == 1


def test_hive_reader_cancels_when_consumer_stops_early(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation(
        [
            _FakeBatch([(1,)], True),
            _FakeBatch([(2,)], False),
        ],
        transport,
    )
    cursor = _FakeCursor(operation)
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )

    iterator = _reader().read(_manifest())
    assert next(iterator).column("id").to_pylist() == [1]
    iterator.close()

    assert cursor.cancel_calls == 1
    assert cursor.close_calls == 1
    assert connection.close_calls == 1


def test_hive_reader_does_not_reexecute_after_a_partial_stream_failure(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation([_FakeBatch([(1,)], True)], transport)
    original_fetch = operation.fetch

    def fail_second_fetch(*args, **kwargs):
        if len(operation.fetch_calls) > 0:
            raise OSError("simulated disconnect after first response")
        return original_fetch(*args, **kwargs)

    operation.fetch = fail_second_fetch
    cursor = _FakeCursor(operation)
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )

    iterator = _reader().read(_manifest())
    assert next(iterator).column("id").to_pylist() == [1]
    with pytest.raises(RuntimeError, match="query was not retried"):
        next(iterator)

    assert cursor.execute_calls == ["SELECT `id` FROM `default`.`t`"]
    assert len(operation.fetch_calls) == 1
    assert cursor.cancel_calls == 1


def test_hive_reader_rejects_multiple_manifest_rows_before_connecting(monkeypatch):
    def fail_if_connected(**kwargs):
        raise AssertionError("an invalid manifest must fail before connecting")

    monkeypatch.setattr(hive_reader, "connect_hiveserver2", fail_if_connected)
    manifest = FileManifest.construct_manifest(
        paths=["hive://planned-read", "hive://planned-read"],
        sizes=[1, 1],
        chunk_metadatas=[None, None],
    )

    with pytest.raises(ValueError, match="one manifest row"):
        list(_reader().read(manifest))


def test_rows_to_arrow_preserves_nanosecond_timestamp_precision():
    schema = pa.schema([("event_time", pa.timestamp("ns"))])
    table = hive_reader._rows_to_arrow(
        [
            ("2026-09-20 12:34:56.123456789",),
            ("1969-12-31 23:59:59.999999999",),
            (None,),
        ],
        schema,
    )

    assert table.column("event_time").cast(pa.int64()).to_pylist() == [
        1_789_907_696_123_456_789,
        -1,
        None,
    ]


def test_hive_reader_rejects_row_protocol_for_timestamps_before_fetch(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation(
        [_FakeBatch([("2026-09-20 12:34:56.123456789",)], False)], transport
    )
    operation.is_columnar = False
    cursor = _FakeCursor(operation)
    cursor.description = [("event_time", "TIMESTAMP", None, None, None, None, None)]
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )
    reader = HiveReader(
        connection_options=_reader().connection_options,
        query="SELECT `event_time` FROM `default`.`t`",
        expected_query_schema=pa.schema([("event_time", pa.timestamp("ns"))]),
        output_schema=pa.schema([("event_time", pa.timestamp("ns"))]),
    )

    with pytest.raises(ValueError, match="require columnar results"):
        list(reader.read(_manifest()))

    assert operation.fetch_calls == []
    assert cursor.cancel_calls == 1


def test_hive_reader_preserves_columnar_timestamp_precision(monkeypatch):
    transport = _FakeMessageTransport()
    operation = _FakeOperation(
        [_FakeBatch([("2026-09-20 12:34:56.123456789",)], False)], transport
    )
    cursor = _FakeCursor(operation)
    cursor.description = [("event_time", "TIMESTAMP", None, None, None, None, None)]
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(
        hive_reader,
        "connect_hiveserver2",
        lambda **kwargs: (connection, transport),
    )
    reader = HiveReader(
        connection_options=_reader().connection_options,
        query="SELECT `event_time` FROM `default`.`t`",
        expected_query_schema=pa.schema([("event_time", pa.timestamp("ns"))]),
        output_schema=pa.schema([("event_time", pa.timestamp("ns"))]),
    )

    tables = list(reader.read(_manifest()))

    assert len(tables) == 1
    assert tables[0].column("event_time").cast(pa.int64()).to_pylist() == [
        1_789_907_696_123_456_789
    ]
    assert cursor.convert_types is False


def test_rows_to_arrow_converts_decimal_strings():
    schema = pa.schema([("price", pa.decimal128(10, 2))])

    table = hive_reader._rows_to_arrow([("123.45",), (None,)], schema)

    assert table.column("price").to_pylist() == [Decimal("123.45"), None]
