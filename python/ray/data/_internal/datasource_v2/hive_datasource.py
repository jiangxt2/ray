"""HiveServer2-backed Ray Data V2 datasource."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, List, Optional, Tuple

import pyarrow as pa
from pyarrow.fs import FileSystem
from typing_extensions import override

from ray.data._internal.datasource.hive_hs2_client import (
    HiveClientCompatibilityError,
    HiveConnectionOptions,
    HiveResponseTooLargeError,
    connect_hiveserver2,
)
from ray.data._internal.datasource_v2.datasource_v2 import (
    DatasourceCategory,
    DataSourceV2,
)
from ray.data._internal.datasource_v2.hive_types import (
    MAX_HIVE_COLUMNS,
    HiveTypeCompatibilityError,
    schema_from_table_columns,
)
from ray.data._internal.datasource_v2.listing.file_indexer import (
    FileIndexer,
    FileInfo,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.partitioners.file_partitioner import (
    FilePartitioner,
)
from ray.data._internal.datasource_v2.readers.hive_reader import (
    DEFAULT_HIVE_TARGET_BATCH_BYTES,
)
from ray.data._internal.datasource_v2.scanners.hive_scanner import HiveScanner

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_OPAQUE_INPUT_IDENTIFIER = "hive://planned-read"


def _escape_hs2_metadata_pattern(identifier: str) -> str:
    """Escape JDBC search-pattern characters in a Hive identifier."""
    return identifier.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class _HiveInputIndexer(FileIndexer):
    """Emit exactly one opaque manifest row; no filesystem listing is done."""

    def list_files(
        self,
        paths,
        *,
        filesystem: Optional[FileSystem],
        pruners=None,
        preserve_order: bool = False,
        predicate=None,
        limit: Optional[int] = None,
        projected_columns=None,
        shuffle_config=None,
        execution_idx: int = 0,
    ) -> Iterable[FileManifest]:
        identifiers = paths.to_pylist()
        if not identifiers:
            return
        if len(identifiers) != 1 or identifiers[0] != _OPAQUE_INPUT_IDENTIFIER:
            raise ValueError("Hive listing expected one opaque input identifier")
        if limit == 0:
            return
        yield FileManifest.construct_manifest(
            paths=[_OPAQUE_INPUT_IDENTIFIER], sizes=[1], chunk_metadatas=[None]
        )

    def list_file_infos(
        self,
        paths,
        *,
        filesystem: Optional[FileSystem],
        pruners=None,
        preserve_order: bool = False,
    ) -> Iterable[FileInfo]:
        identifiers = paths.to_pylist()
        if not identifiers:
            return
        if len(identifiers) != 1 or identifiers[0] != _OPAQUE_INPUT_IDENTIFIER:
            raise ValueError("Hive listing expected one opaque input identifier")
        yield FileInfo(path=_OPAQUE_INPUT_IDENTIFIER, size=1)


class _SingleHiveInputPartitioner(FilePartitioner):
    """Keep the opaque database input as one indivisible read unit."""

    def __init__(self):
        self._manifest: Optional[FileManifest] = None

    @property
    def requires_global_input(self) -> bool:
        return True

    def add_input(self, input_manifest: FileManifest) -> None:
        if len(input_manifest) == 0:
            return
        self._manifest = (
            input_manifest
            if self._manifest is None
            else FileManifest.concat([self._manifest, input_manifest])
        )

    def has_partition(self) -> bool:
        return self._manifest is not None

    def next_partition(self) -> FileManifest:
        if self._manifest is None:
            raise StopIteration
        manifest, self._manifest = self._manifest, None
        return manifest

    def finalize(self) -> None:
        pass


class HiveDatasourceV2(DataSourceV2[FileManifest]):
    """A one-operation HS2 datasource using Ray Data's V2 scanner/reader path."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        connection_options: HiveConnectionOptions,
        table: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[pa.Schema] = None,
        source_limit: Optional[int] = None,
    ):
        super().__init__(name="HiveV2", category=DatasourceCategory.DATABASE)
        if (table is None) == (query is None):
            raise ValueError("Specify exactly one of table or query")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        if source_limit is not None and (
            isinstance(source_limit, bool)
            or not isinstance(source_limit, int)
            or source_limit < 0
        ):
            raise ValueError("limit must be a non-negative integer")
        if connection_options.host != host or connection_options.port != port:
            raise ValueError(
                "Hive connection options do not match the datasource endpoint"
            )
        if not isinstance(connection_options.auth_mechanism, str):
            raise ValueError("auth_mechanism must be a string")
        mechanism = connection_options.auth_mechanism.upper()
        if mechanism not in {"NOSASL", "PLAIN", "GSSAPI"}:
            raise ValueError("auth_mechanism must be NOSASL, PLAIN, or GSSAPI")
        if connection_options.user is not None and (
            not isinstance(connection_options.user, str) or not connection_options.user
        ):
            raise ValueError("user must be a non-empty string when provided")
        if connection_options.password is not None and not isinstance(
            connection_options.password, str
        ):
            raise ValueError("password must be a string when provided")
        if mechanism == "PLAIN" and (
            not connection_options.user or not connection_options.password
        ):
            raise ValueError(
                "PLAIN authentication requires a user and non-empty password"
            )
        if connection_options.ca_cert is not None and not isinstance(
            connection_options.ca_cert, str
        ):
            raise ValueError("ca_cert must be a string when provided")
        if connection_options.ca_cert and not connection_options.use_ssl:
            raise ValueError("ca_cert requires use_ssl=True")
        if not isinstance(connection_options.use_ssl, bool):
            raise ValueError("use_ssl must be a bool")
        if connection_options.timeout is not None and (
            isinstance(connection_options.timeout, bool)
            or not isinstance(connection_options.timeout, (int, float))
            or not math.isfinite(connection_options.timeout)
            or connection_options.timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")

        self._connection_options = connection_options
        self._table_mode = table is not None
        self._source_limit = source_limit
        self._table_name: Optional[str] = None
        self._database_name: Optional[str] = None
        self._query: str

        if table is not None:
            if query is not None or schema is not None:
                raise ValueError("schema and query are only valid for raw SQL reads")
            self._database_name, self._table_name = self._parse_table_identifier(table)
            self._query = (
                f"{self._quote_identifier(self._database_name)}."
                f"{self._quote_identifier(self._table_name)}"
            )
            self._schema = self._resolve_table_schema()
        else:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty SQL string")
            if not isinstance(schema, pa.Schema):
                raise ValueError("raw SQL reads require an explicit pyarrow.Schema")
            if len(schema) > MAX_HIVE_COLUMNS or len(set(schema.names)) != len(
                schema.names
            ):
                raise ValueError("raw SQL schema has too many or duplicate columns")
            if source_limit is not None:
                raise ValueError("limit is only supported for table-oriented reads")
            self._query = query
            self._schema = schema

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return "`" + identifier.replace("`", "``") + "`"

    @classmethod
    def _parse_table_identifier(cls, table: str) -> Tuple[str, str]:
        if not isinstance(table, str):
            raise ValueError("table must be a string")
        parts = table.split(".")
        if len(parts) == 1:
            parts.insert(0, "default")
        if len(parts) != 2 or any(not _IDENTIFIER_RE.fullmatch(part) for part in parts):
            raise ValueError(
                "table must be a simple Hive identifier, optionally qualified "
                "by database"
            )
        return parts[0], parts[1]

    def _resolve_table_schema(self) -> pa.Schema:
        connection = None
        cursor = None
        operation = None
        message_transport = None
        response_too_large = False
        try:
            connection, message_transport = connect_hiveserver2(
                host=self._connection_options.host,
                port=self._connection_options.port,
                auth_mechanism=self._connection_options.auth_mechanism,
                user=self._connection_options.user,
                password=self._connection_options.password,
                kerberos_service_name=self._connection_options.kerberos_service_name,
                use_ssl=self._connection_options.use_ssl,
                ca_cert=self._connection_options.ca_cert,
                timeout=self._connection_options.timeout,
            )
            cursor = connection.cursor(user=self._connection_options.user)
            # These are HiveServer2 metadata RPCs, not SQL data queries.
            database_pattern = _escape_hs2_metadata_pattern(self._database_name)
            table_pattern = _escape_hs2_metadata_pattern(self._table_name)
            operation = cursor.session.get_tables(
                database=database_pattern,
                table_like=table_pattern,
            )
            matches = []
            empty_batches = 0
            while True:
                batch = operation.fetch(max_rows=16)
                if batch is None:
                    break
                expect_more_rows = bool(getattr(batch, "expect_more_rows", False))
                rows = batch.pop_many(len(batch))
                if not rows:
                    if not expect_more_rows:
                        break
                    empty_batches += 1
                    if empty_batches > 100:
                        raise RuntimeError(
                            "HiveServer2 returned too many empty table batches"
                        )
                    continue
                empty_batches = 0
                for row in rows:
                    if len(row) > 3:
                        row_database = str(row[1]).casefold()
                        row_table = str(row[2]).casefold()
                        if (
                            row_database == self._database_name.casefold()
                            and row_table == self._table_name.casefold()
                        ):
                            matches.append(row)
            if not matches:
                raise ValueError(
                    f"Hive relation {self._database_name}.{self._table_name} "
                    "was not found or is not visible to this identity"
                )
            if len(matches) > 1:
                raise HiveTypeCompatibilityError(
                    "Hive metadata returned multiple matches for the requested relation"
                )
            supported_kinds = {
                "TABLE",
                "VIEW",
                "MANAGED_TABLE",
                "EXTERNAL_TABLE",
                "VIRTUAL_VIEW",
                "MATERIALIZED_VIEW",
            }
            if not any(str(row[3]).upper() in supported_kinds for row in matches):
                raise ValueError("Hive relation has an unsupported relation type")
            operation.close()
            operation = None

            operation = cursor.session.get_table_schema(table_pattern, database_pattern)
            description = operation.get_result_schema()
            columns = []
            empty_batches = 0
            while True:
                batch = operation.fetch(
                    schema=description,
                    max_rows=min(256, MAX_HIVE_COLUMNS + 1 - len(columns)),
                    convert_types=True,
                    convert_strings_to_unicode=True,
                )
                if batch is None:
                    break
                expect_more_rows = bool(getattr(batch, "expect_more_rows", False))
                rows = batch.pop_many(len(batch))
                if not rows:
                    if not expect_more_rows:
                        break
                    empty_batches += 1
                    if empty_batches > 100:
                        raise RuntimeError(
                            "HiveServer2 returned too many empty schema batches"
                        )
                    continue
                empty_batches = 0
                for row in rows:
                    if len(row) < 6:
                        raise HiveTypeCompatibilityError(
                            "Hive GetColumns metadata has an unexpected shape"
                        )
                    row_database = str(row[1]).casefold()
                    row_table = str(row[2]).casefold()
                    if (
                        row_database != self._database_name.casefold()
                        or row_table != self._table_name.casefold()
                    ):
                        raise HiveTypeCompatibilityError(
                            "Hive GetColumns metadata included columns for a "
                            "different relation"
                        )
                    name = str(row[3])
                    type_name = str(row[5])
                    if type_name.strip().upper() == "DECIMAL":
                        precision = row[6] if len(row) > 6 else None
                        scale = row[8] if len(row) > 8 else None
                        if precision is None or scale is None:
                            raise HiveTypeCompatibilityError(
                                f"Hive decimal column {name!r} is missing "
                                "precision/scale"
                            )
                        type_name = f"DECIMAL({int(precision)},{int(scale)})"
                    columns.append((name, type_name))
                    if len(columns) > MAX_HIVE_COLUMNS:
                        raise HiveTypeCompatibilityError(
                            "Hive relation has more than the supported "
                            f"{MAX_HIVE_COLUMNS} columns"
                        )
                # HS2's hasMoreRows flag is optional and Impyla maps an
                # omitted value to False. A non-empty page is not an end of
                # stream; fetch until an empty page reports no more rows.
            return schema_from_table_columns(columns)
        except (ValueError, HiveClientCompatibilityError):
            raise
        except HiveResponseTooLargeError:
            response_too_large = True
            if cursor is not None and hasattr(cursor, "_closed"):
                cursor._closed = True
            if message_transport is not None:
                try:
                    message_transport.close()
                except Exception:
                    pass
            raise
        except Exception as exc:
            raise RuntimeError(
                f"HiveServer2 planning failed ({type(exc).__name__})"
            ) from None
        finally:
            if not response_too_large:
                if operation is not None:
                    try:
                        operation.close()
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

    @property
    @override
    def paths(self) -> List[str]:
        return [_OPAQUE_INPUT_IDENTIFIER]

    @property
    @override
    def filesystem(self) -> Optional[FileSystem]:
        return None

    @override
    def _get_file_indexer(self) -> FileIndexer:
        return _HiveInputIndexer()

    @override
    def get_file_partitioner(self, **kwargs: Any) -> FilePartitioner:
        # The manifest carries one opaque query token, not files whose sizes
        # can estimate or split the HS2 result stream.
        return _SingleHiveInputPartitioner()

    @property
    @override
    def schema_needs_file_sample(self) -> bool:
        return False

    @override
    def infer_schema(self, sample: Optional[FileManifest]) -> pa.Schema:
        if sample is not None:
            raise ValueError("Hive schema is resolved from HS2 planning metadata")
        return self._schema

    @override
    def create_scanner(
        self,
        schema: pa.Schema,
        filesystem: Optional[FileSystem] = None,
        **options: Any,
    ) -> HiveScanner:
        if filesystem is not None:
            raise ValueError("HiveServer2 reads do not use a Ray filesystem")
        target_block_size = options.get(
            "target_max_block_size", DEFAULT_HIVE_TARGET_BATCH_BYTES
        )
        if target_block_size is None:
            target_block_size = DEFAULT_HIVE_TARGET_BATCH_BYTES
        target_batch_bytes = max(
            1, min(DEFAULT_HIVE_TARGET_BATCH_BYTES, target_block_size)
        )
        return HiveScanner(
            schema=schema,
            connection_options=self._connection_options,
            base_query=self._query,
            table_mode=self._table_mode,
            source_limit=self._source_limit,
            target_batch_bytes=target_batch_bytes,
        )
