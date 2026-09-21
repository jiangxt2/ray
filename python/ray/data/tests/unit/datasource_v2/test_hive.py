import struct
from decimal import Decimal

import pyarrow as pa
import pytest

from ray.data._internal.datasource.hive_hs2_client import (
    MAX_HS2_RESPONSE_BYTES,
    HiveClientCompatibilityError,
    HiveResponseTooLargeError,
    connect_hiveserver2,
    _make_bounded_binary_protocol,
    _make_bounded_sasl_transport,
    _MessageBudgetTransport,
)
from ray.data._internal.datasource_v2.hive_datasource import (
    _OPAQUE_INPUT_IDENTIFIER,
    _HiveInputIndexer,
    _escape_hs2_metadata_pattern,
)
from ray.data._internal.datasource_v2.hive_sql import (
    quote_relation,
    split_predicate,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.hive_types import (
    HiveTypeCompatibilityError,
    hive_type_to_arrow,
    schema_from_description,
)
from ray.data._internal.datasource_v2.readers.hive_reader import _AdaptiveBatchSizer
from ray.data._internal.datasource_v2.scanners.hive_scanner import HiveScanner
from ray.data.expressions import col, lit


class _FakeTransport:
    def __init__(self, data=b"abcdef"):
        self.data = data
        self.read_all_calls = []

    def readAll(self, size):
        self.read_all_calls.append(size)
        result, self.data = self.data[:size], self.data[size:]
        if len(result) != size:
            raise EOFError
        return result

    def read(self, size):
        return self.readAll(size)

    def write(self, data):
        pass

    def flush(self):
        pass

    def isOpen(self):
        return True

    def open(self):
        pass

    def close(self):
        pass


def test_hive_schema_maps_supported_primitive_and_temporal_types():
    schema = schema_from_description(
        [
            ("enabled", "BOOLEAN"),
            ("tiny", "TINYINT"),
            ("name", "VARCHAR(64)"),
            ("event_date", "DATE"),
            ("event_time", "TIMESTAMP"),
            ("price", "DECIMAL(10,2)"),
        ]
    )
    assert schema.types == [
        pa.bool_(),
        pa.int8(),
        pa.string(),
        pa.date32(),
        pa.timestamp("ns"),
        pa.decimal128(10, 2),
    ]


def test_hive_schema_rejects_ambiguous_and_complex_types():
    with pytest.raises(HiveTypeCompatibilityError, match="unsupported or ambiguous"):
        hive_type_to_arrow("DECIMAL", "price")
    with pytest.raises(HiveTypeCompatibilityError, match="unsupported or ambiguous"):
        hive_type_to_arrow("ARRAY<STRING>", "tags")


def test_hive_schema_rejects_duplicate_names():
    with pytest.raises(HiveTypeCompatibilityError, match="duplicate"):
        schema_from_description([("id", "INT"), ("id", "BIGINT")])


def test_table_identifier_quoting_escapes_backticks():
    assert quote_relation("db`name", "table") == "`db``name`.`table`"


def test_hs2_metadata_patterns_escape_wildcards():
    assert _escape_hs2_metadata_pattern("db_test") == r"db\_test"
    assert _escape_hs2_metadata_pattern("table%name") == r"table\%name"
    assert _escape_hs2_metadata_pattern(r"table\name") == r"table\\name"


def test_predicate_pushdown_keeps_unsupported_conjuncts_as_residuals():
    schema = pa.schema([("age", pa.int64()), ("name", pa.string())])
    predicate = (col("age") >= 18) & (col("name") == "Ray") & (col("age") + 1 > 20)

    sql, residual = split_predicate(predicate, schema)

    assert sql == "((`age` >= 18) AND (`name` = 'Ray'))"
    assert residual is not None


@pytest.mark.parametrize("value", ["O'Reilly", r"back\\slash", "line\nbreak"])
def test_strings_with_sql_escape_sensitive_characters_stay_residual(value):
    sql, residual = split_predicate(
        col("name") == lit(value), pa.schema([("name", pa.string())])
    )

    assert sql is None
    assert residual is not None


def test_timestamp_pushdown_preserves_nanosecond_precision():
    pd = pytest.importorskip("pandas")
    value = pd.Timestamp("2026-09-20 12:34:56.123456789")

    sql, residual = split_predicate(
        col("event_time") >= lit(value),
        pa.schema([("event_time", pa.timestamp("ns"))]),
    )

    assert sql == "`event_time` >= TIMESTAMP '2026-09-20 12:34:56.123456789'"
    assert residual is None


def test_non_finite_decimal_literal_stays_as_residual():
    sql, residual = split_predicate(
        col("price") == lit(Decimal("NaN")),
        pa.schema([("price", pa.decimal128(10, 2))]),
    )

    assert sql is None
    assert residual is not None


def test_nullable_comparison_is_not_rewritten_as_is_null():
    sql, residual = split_predicate(
        col("name") == lit(None), pa.schema([("name", pa.string())])
    )

    assert sql is None
    assert residual is not None


def test_hive_indexer_emits_exactly_one_opaque_manifest_row():
    indexer = _HiveInputIndexer()
    manifests = list(
        indexer.list_files(pa.array([_OPAQUE_INPUT_IDENTIFIER]), filesystem=None)
    )
    assert len(manifests) == 1
    assert len(manifests[0]) == 1
    assert manifests[0].paths.tolist() == [_OPAQUE_INPUT_IDENTIFIER]


def test_hive_indexer_rejects_multiple_input_identifiers():
    indexer = _HiveInputIndexer()
    with pytest.raises(ValueError, match="one opaque input"):
        list(
            indexer.list_files(
                pa.array([_OPAQUE_INPUT_IDENTIFIER, _OPAQUE_INPUT_IDENTIFIER]),
                filesystem=None,
            )
        )


def test_message_budget_transport_accepts_a_response_at_the_limit():
    transport = _FakeTransport(data=b"abcd")
    budget = _MessageBudgetTransport(transport, max_bytes=4)

    assert budget.readAll(4) == b"abcd"
    assert budget.last_message_bytes == 4


def test_message_budget_transport_rejects_before_reading_over_limit():
    transport = _FakeTransport(data=b"abcdef")
    budget = _MessageBudgetTransport(transport, max_bytes=4)

    with pytest.raises(HiveResponseTooLargeError):
        budget.readAll(5)

    assert transport.read_all_calls == []
    assert budget.last_message_bytes == 0


def test_message_budget_transport_resets_at_each_rpc():
    transport = _FakeTransport(data=b"abcdefgh")
    budget = _MessageBudgetTransport(transport, max_bytes=4)

    assert budget.readAll(4) == b"abcd"
    budget.begin_message()
    assert budget.readAll(4) == b"efgh"
    assert budget.last_message_bytes == 4


def test_adaptive_batch_sizer_starts_small_then_uses_observed_bytes():
    sizer = _AdaptiveBatchSizer(target_bytes=1024, column_count=1)
    small_table = pa.table({"value": list(range(8))})

    next_rows = sizer.observe(small_table, wire_bytes=128)

    assert next_rows > 1
    assert next_rows <= 512


def test_response_limit_is_bounded_and_internal():
    assert MAX_HS2_RESPONSE_BYTES == 1024 * 1024


def test_binary_protocol_rejects_declared_string_before_reading_payload():
    pytest.importorskip("thrift")
    transport = _FakeTransport(data=struct.pack("!i", MAX_HS2_RESPONSE_BYTES + 1))
    protocol = _make_bounded_binary_protocol(
        _MessageBudgetTransport(transport, MAX_HS2_RESPONSE_BYTES)
    )

    with pytest.raises(HiveResponseTooLargeError, match="protocol limit"):
        protocol.readString()

    assert transport.read_all_calls == [4]


def test_binary_protocol_rejects_declared_container_before_reading_payload():
    pytest.importorskip("thrift")
    transport = _FakeTransport(data=b"\x08" + struct.pack("!i", 65_537))
    protocol = _make_bounded_binary_protocol(
        _MessageBudgetTransport(transport, MAX_HS2_RESPONSE_BYTES)
    )

    with pytest.raises(HiveResponseTooLargeError, match="protocol limit"):
        protocol.readListBegin()

    assert transport.read_all_calls == [1, 4]


def test_sasl_transport_rejects_frame_before_reading_payload():
    pytest.importorskip("thrift_sasl")

    class _SaslSocket(_FakeTransport):
        def __init__(self):
            super().__init__(data=struct.pack(">I", MAX_HS2_RESPONSE_BYTES + 1))

    socket = _SaslSocket()
    transport = _make_bounded_sasl_transport(lambda: None, "PLAIN", socket)

    with pytest.raises(HiveResponseTooLargeError, match="SASL frame"):
        transport._read_frame()

    assert socket.read_all_calls == [4]


def test_adaptive_batch_sizer_accounts_for_python_row_width():
    sizer = _AdaptiveBatchSizer(target_bytes=1024 * 1024, column_count=8192)
    empty_width_table = pa.table({f"c{i}": [None] for i in range(8192)})

    assert sizer.observe(empty_width_table, wire_bytes=0) == 1


def test_hive_scanner_aliases_columns_to_unqualified_result_names():
    scanner = HiveScanner(
        schema=pa.schema([("id", pa.int32()), ("value", pa.string())]),
        connection_options=None,
        base_query="`default`.`users`",
        table_mode=True,
    )

    assert scanner._build_query() == (
        "SELECT `id` AS `id`, `value` AS `value` FROM `default`.`users`"
    )
    assert scanner.prune_columns(["value"])._build_query() == (
        "SELECT `value` AS `value` FROM `default`.`users`"
    )


class _MetadataBatch:
    def __init__(self, rows, expect_more_rows=False):
        self.rows = list(rows)
        self.expect_more_rows = expect_more_rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def pop_many(self, count):
        result, self.rows = self.rows[:count], self.rows[count:]
        return result


class _MetadataOperation:
    def __init__(self, rows):
        self.rows = list(rows)
        self.fetch_calls = []
        self.close_calls = 0

    def fetch(self, *args, **kwargs):
        self.fetch_calls.append(kwargs)
        max_rows = kwargs.get("max_rows", len(self.rows))
        rows, self.rows = self.rows[:max_rows], self.rows[max_rows:]
        return _MetadataBatch(rows, expect_more_rows=False)

    def get_result_schema(self):
        return [("metadata", "STRING")]

    def close(self):
        self.close_calls += 1


class _MetadataSession:
    def __init__(self, table_operation, columns_operation):
        self.table_operation = table_operation
        self.columns_operation = columns_operation
        self.table_request = None
        self.columns_request = None

    def get_tables(self, **kwargs):
        self.table_request = kwargs
        return self.table_operation

    def get_table_schema(self, table, database):
        self.columns_request = (table, database)
        return self.columns_operation


class _MetadataCursor:
    def __init__(self, session):
        self.session = session
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _MetadataConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.close_calls = 0

    def cursor(self, user=None):
        return self._cursor

    def close(self):
        self.close_calls += 1


@pytest.mark.parametrize("relation_type", ["VIEW", "MATERIALIZED_VIEW"])
def test_hive_table_planning_uses_only_metadata_rpcs(monkeypatch, relation_type):
    from ray.data._internal.datasource import hive_hs2_client
    from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2

    database_name = "db_test"
    table_name = "user_table"
    table_rows = [
        (None, database_name, "userXtable", "TABLE", "", "") for _ in range(8)
    ]
    table_rows.extend(
        (None, f"db{i}test", table_name, "TABLE", "", "") for i in range(8)
    )
    table_rows.append((None, database_name, table_name, relation_type, "", ""))
    table_operation = _MetadataOperation(table_rows)
    column_rows = [
        (None, database_name, table_name, "id", 4, "INT", 10, 4, 0, 10, 1),
        (None, database_name, table_name, "price", 3, "DECIMAL", 10, 0, 2, 10, 1),
    ]
    column_rows.extend(
        (None, database_name, table_name, f"col{i}", 4, "INT", 10, 4, 0, 10, 1)
        for i in range(2, 257)
    )
    columns_operation = _MetadataOperation(column_rows)
    session = _MetadataSession(table_operation, columns_operation)
    cursor = _MetadataCursor(session)
    connection = _MetadataConnection(cursor)
    monkeypatch.setattr(
        "ray.data._internal.datasource_v2.hive_datasource.connect_hiveserver2",
        lambda **kwargs: (connection, object()),
    )

    datasource = object.__new__(HiveDatasourceV2)
    datasource._connection_options = hive_hs2_client.HiveConnectionOptions(
        host="hive-server", port=10000, auth_mechanism="NOSASL"
    )
    datasource._database_name = database_name
    datasource._table_name = table_name

    schema = datasource._resolve_table_schema()

    assert len(schema) == 257
    assert schema.names[:2] == ["id", "price"]
    assert schema.types[:2] == [pa.int32(), pa.decimal128(10, 2)]
    assert session.table_request == {
        "database": r"db\_test",
        "table_like": r"user\_table",
    }
    assert session.columns_request == (r"user\_table", r"db\_test")
    assert len(table_operation.fetch_calls) == 3
    assert len(columns_operation.fetch_calls) == 3
    assert cursor.close_calls == 1
    assert connection.close_calls == 1
    assert columns_operation.close_calls == 1


def test_hive_table_planning_rejects_columns_from_another_relation(monkeypatch):
    from ray.data._internal.datasource import hive_hs2_client
    from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2

    table_operation = _MetadataOperation([(None, "default", "users", "TABLE", "", "")])
    columns_operation = _MetadataOperation(
        [(None, "other_db", "users", "foreign_column", 4, "INT", 10, 4, 0, 10, 1)]
    )
    session = _MetadataSession(table_operation, columns_operation)
    cursor = _MetadataCursor(session)
    connection = _MetadataConnection(cursor)
    monkeypatch.setattr(
        "ray.data._internal.datasource_v2.hive_datasource.connect_hiveserver2",
        lambda **kwargs: (connection, object()),
    )

    datasource = object.__new__(HiveDatasourceV2)
    datasource._connection_options = hive_hs2_client.HiveConnectionOptions(
        host="hive-server", port=10000, auth_mechanism="NOSASL"
    )
    datasource._database_name = "default"
    datasource._table_name = "users"

    with pytest.raises(HiveTypeCompatibilityError, match="different relation"):
        datasource._resolve_table_schema()

    assert columns_operation.close_calls == 1
    assert cursor.close_calls == 1
    assert connection.close_calls == 1


@pytest.mark.parametrize(
    ("package", "unsupported_version"),
    [("impyla", "0.25.0"), ("thrift", "0.25.0"), ("thrift-sasl", "0.4.4")],
)
def test_connect_hiveserver2_rejects_unsupported_hs2_stack_version(
    monkeypatch, package, unsupported_version
):
    from ray.data._internal.datasource import hive_hs2_client

    supported_versions = {
        "impyla": "0.24.0",
        "thrift": "0.24.0",
        "thrift-sasl": "0.4.3",
    }
    supported_versions[package] = unsupported_version
    monkeypatch.setattr(
        hive_hs2_client, "version", lambda name: supported_versions[name]
    )

    with pytest.raises(HiveClientCompatibilityError, match=f"supports {package}=="):
        connect_hiveserver2(
            host="hive-server",
            port=10000,
            auth_mechanism="NOSASL",
            user=None,
            password=None,
            kerberos_service_name="hive",
            use_ssl=False,
            ca_cert=None,
            timeout=None,
        )


def test_hive_planning_preserves_impyla_compatibility_error(monkeypatch):
    from ray.data._internal.datasource import hive_hs2_client
    from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2

    monkeypatch.setattr(
        "ray.data._internal.datasource_v2.hive_datasource.connect_hiveserver2",
        lambda **kwargs: (_ for _ in ()).throw(
            HiveClientCompatibilityError("unsupported Impyla version")
        ),
    )
    datasource = object.__new__(HiveDatasourceV2)
    datasource._connection_options = hive_hs2_client.HiveConnectionOptions(
        host="hive-server", port=10000, auth_mechanism="NOSASL"
    )
    datasource._database_name = "default"
    datasource._table_name = "users"

    with pytest.raises(HiveClientCompatibilityError, match="unsupported Impyla"):
        datasource._resolve_table_schema()


def test_connect_hiveserver2_enables_tls_peer_verification(monkeypatch):
    impyla_thrift_api = pytest.importorskip("impala._thrift_api")
    from ray.data._internal.datasource.hive_hs2_client import connect_hiveserver2

    class _FailingSocket:
        def isOpen(self):
            return False

        def open(self):
            raise OSError("stop before network")

        def close(self):
            pass

    received = {}

    def fake_get_socket(host, port, **kwargs):
        received["host"] = host
        received["port"] = port
        received.update(kwargs)
        return _FailingSocket()

    monkeypatch.setattr(impyla_thrift_api, "get_socket", fake_get_socket)

    with pytest.raises(OSError, match="stop before network"):
        connect_hiveserver2(
            host="hive-server",
            port=10000,
            auth_mechanism="NOSASL",
            user=None,
            password=None,
            kerberos_service_name="hive",
            use_ssl=True,
            ca_cert="/tmp/hive-ca.pem",
            timeout=None,
        )

    assert received == {
        "host": "hive-server",
        "port": 10000,
        "use_ssl": True,
        "ca_cert": "/tmp/hive-ca.pem",
        "verify_cert": True,
    }


def test_raw_sql_schema_column_limit_fails_before_execution():
    from ray.data._internal.datasource.hive_hs2_client import HiveConnectionOptions
    from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2

    oversized_schema = pa.schema([pa.field(f"c{i}", pa.int8()) for i in range(8193)])
    options = HiveConnectionOptions(
        host="hive-server", port=10000, auth_mechanism="NOSASL"
    )
    with pytest.raises(ValueError, match="too many or duplicate columns"):
        HiveDatasourceV2(
            host="hive-server",
            port=10000,
            connection_options=options,
            query="SELECT 1",
            schema=oversized_schema,
        )


def test_hive_partitioner_preserves_one_indivisible_query_unit():
    from ray.data._internal.datasource_v2.hive_datasource import (
        _SingleHiveInputPartitioner,
    )

    manifest = FileManifest.construct_manifest(
        paths=[_OPAQUE_INPUT_IDENTIFIER], sizes=[1], chunk_metadatas=[None]
    )
    partitioner = _SingleHiveInputPartitioner()
    partitioner.add_input(manifest)
    partitioner.add_input(manifest)
    partitioner.finalize()

    assert partitioner.requires_global_input is True
    assert partitioner.has_partition()
    assert len(partitioner.next_partition()) == 2
    assert not partitioner.has_partition()


def test_hive_datasource_rejects_empty_plain_password():
    from ray.data._internal.datasource.hive_hs2_client import HiveConnectionOptions
    from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2

    options = HiveConnectionOptions(
        host="hive-server",
        port=10000,
        auth_mechanism="PLAIN",
        user="ray",
        password="",
    )
    with pytest.raises(ValueError, match="non-empty password"):
        HiveDatasourceV2(
            host="hive-server",
            port=10000,
            connection_options=options,
            query="SELECT 1",
            schema=pa.schema([("value", pa.int32())]),
        )
