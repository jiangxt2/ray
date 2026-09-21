"""Opt-in real-Hive integration test for the V2 HiveServer2 datasource."""

import os
import uuid

import pyarrow as pa
import pytest

import ray
from ray.data._internal.datasource.hive_hs2_client import (
    HiveConnectionOptions,
    connect_hiveserver2,
)

_HOST = os.environ.get("RAY_DATA_HIVE_TEST_HOST")
_PORT = int(os.environ.get("RAY_DATA_HIVE_TEST_PORT", "10000"))
_AUTH = os.environ.get("RAY_DATA_HIVE_TEST_AUTH", "PLAIN")
_USER = os.environ.get("RAY_DATA_HIVE_TEST_USER", "ray")
_PASSWORD = os.environ.get("HIVE_TEST_PASSWORD")
_KERBEROS_SERVICE = os.environ.get("RAY_DATA_HIVE_TEST_KERBEROS_SERVICE", "hive")
_TEST_MATERIALIZED_VIEW = os.environ.get("RAY_DATA_HIVE_TEST_MATERIALIZED_VIEW") == "1"

pytestmark = [
    pytest.mark.data_integration,
    pytest.mark.skipif(not _HOST, reason="RAY_DATA_HIVE_TEST_HOST is not configured"),
]


def test_read_hive_table_and_raw_sql_over_real_hs2():
    if _AUTH.upper() == "PLAIN" and not _PASSWORD:
        pytest.skip("PLAIN integration requires HIVE_TEST_PASSWORD")
    options = HiveConnectionOptions(
        host=_HOST,
        port=_PORT,
        auth_mechanism=_AUTH,
        user=_USER,
        password=_PASSWORD,
        kerberos_service_name=_KERBEROS_SERVICE,
    )
    connection, _ = connect_hiveserver2(
        host=options.host,
        port=options.port,
        auth_mechanism=options.auth_mechanism,
        user=options.user,
        password=options.password,
        kerberos_service_name=options.kerberos_service_name,
        use_ssl=options.use_ssl,
        ca_cert=options.ca_cert,
        timeout=options.timeout,
    )
    table = f"ray_hive_{uuid.uuid4().hex}"
    qualified = f"default.{table}"
    created = False
    try:
        cursor = connection.cursor(user=_USER)
        try:
            cursor.execute(
                f"CREATE TABLE `default`.`{table}` (id INT, value STRING) STORED AS ORC"
            )
            created = True
            cursor.execute(
                f"INSERT INTO TABLE `default`.`{table}` VALUES "
                "(1, 'one'), (2, 'two'), (3, 'three')"
            )
            cursor.execute(f"SELECT COUNT(*) FROM `default`.`{table}`")
            assert cursor.fetchone()[0] == 3
        finally:
            cursor.close()
            connection.close()

        dataset = ray.data.read_hive(
            qualified,
            host=_HOST,
            port=_PORT,
            auth_mechanism=_AUTH,
            user=_USER,
            password=_PASSWORD,
            kerberos_service_name=_KERBEROS_SERVICE,
        )
        rows = sorted(dataset.take_all(), key=lambda row: row["id"])
        assert rows == [
            {"id": 1, "value": "one"},
            {"id": 2, "value": "two"},
            {"id": 3, "value": "three"},
        ]

        raw = ray.data.read_hive(
            host=_HOST,
            port=_PORT,
            auth_mechanism=_AUTH,
            user=_USER,
            password=_PASSWORD,
            kerberos_service_name=_KERBEROS_SERVICE,
            query=f"SELECT id, value FROM {qualified} WHERE id > 100",
            schema=pa.schema([("id", pa.int32()), ("value", pa.string())]),
        )
        assert raw.take_all() == []
    finally:
        if created:
            cleanup, _ = connect_hiveserver2(
                host=options.host,
                port=options.port,
                auth_mechanism=options.auth_mechanism,
                user=options.user,
                password=options.password,
                kerberos_service_name=options.kerberos_service_name,
                use_ssl=options.use_ssl,
                ca_cert=options.ca_cert,
                timeout=options.timeout,
            )
            cleanup_cursor = cleanup.cursor(user=_USER)
            try:
                cleanup_cursor.execute(f"DROP TABLE IF EXISTS `default`.`{table}`")
            finally:
                cleanup_cursor.close()
                cleanup.close()


def test_read_hive_materialized_view_with_pattern_like_sibling_over_real_hs2():
    if not _TEST_MATERIALIZED_VIEW:
        pytest.skip("set RAY_DATA_HIVE_TEST_MATERIALIZED_VIEW=1 to test Hive MVs")
    if _AUTH.upper() == "PLAIN" and not _PASSWORD:
        pytest.skip("PLAIN integration requires HIVE_TEST_PASSWORD")

    options = HiveConnectionOptions(
        host=_HOST,
        port=_PORT,
        auth_mechanism=_AUTH,
        user=_USER,
        password=_PASSWORD,
        kerberos_service_name=_KERBEROS_SERVICE,
    )
    source_table = f"ray_hive_source_{uuid.uuid4().hex}"
    materialized_view = f"ray_hive_mv_{uuid.uuid4().hex}"
    sibling_table = materialized_view.replace("_", "X", 1)
    created_source = False
    created_sibling = False
    created_view = False

    connection, _ = connect_hiveserver2(
        host=options.host,
        port=options.port,
        auth_mechanism=options.auth_mechanism,
        user=options.user,
        password=options.password,
        kerberos_service_name=options.kerberos_service_name,
        use_ssl=options.use_ssl,
        ca_cert=options.ca_cert,
        timeout=options.timeout,
    )
    try:
        cursor = connection.cursor(user=_USER)
        try:
            cursor.execute(
                f"CREATE TABLE `default`.`{source_table}` "
                "(id INT, value STRING) STORED AS ORC"
            )
            created_source = True
            cursor.execute(
                f"INSERT INTO TABLE `default`.`{source_table}` VALUES "
                "(1, 'one'), (2, 'two'), (3, 'three')"
            )
            cursor.execute(
                f"CREATE TABLE `default`.`{sibling_table}` (sibling_only INT) "
                "STORED AS ORC"
            )
            created_sibling = True
            cursor.execute(
                f"CREATE MATERIALIZED VIEW `default`.`{materialized_view}` "
                "DISABLE REWRITE AS "
                f"SELECT id, value FROM `default`.`{source_table}`"
            )
            created_view = True
            cursor.execute(f"SELECT COUNT(*) FROM `default`.`{materialized_view}`")
            assert cursor.fetchone()[0] == 3
        finally:
            cursor.close()
            connection.close()

        dataset = ray.data.read_hive(
            f"default.{materialized_view}",
            host=_HOST,
            port=_PORT,
            auth_mechanism=_AUTH,
            user=_USER,
            password=_PASSWORD,
            kerberos_service_name=_KERBEROS_SERVICE,
        )
        rows = sorted(dataset.take_all(), key=lambda row: row["id"])
        assert rows == [
            {"id": 1, "value": "one"},
            {"id": 2, "value": "two"},
            {"id": 3, "value": "three"},
        ]
    finally:
        if created_source or created_sibling or created_view:
            cleanup, _ = connect_hiveserver2(
                host=options.host,
                port=options.port,
                auth_mechanism=options.auth_mechanism,
                user=options.user,
                password=options.password,
                kerberos_service_name=options.kerberos_service_name,
                use_ssl=options.use_ssl,
                ca_cert=options.ca_cert,
                timeout=options.timeout,
            )
            cleanup_cursor = cleanup.cursor(user=_USER)
            try:
                if created_view:
                    cleanup_cursor.execute(
                        "DROP MATERIALIZED VIEW IF EXISTS "
                        f"`default`.`{materialized_view}`"
                    )
                if created_sibling:
                    cleanup_cursor.execute(
                        f"DROP TABLE IF EXISTS `default`.`{sibling_table}`"
                    )
                if created_source:
                    cleanup_cursor.execute(
                        f"DROP TABLE IF EXISTS `default`.`{source_table}`"
                    )
            finally:
                cleanup_cursor.close()
                cleanup.close()
