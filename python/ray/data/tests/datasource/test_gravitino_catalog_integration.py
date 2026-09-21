"""Manual integration tests for a configured Apache Gravitino 1.3.0 service.

Run this target only against a disposable integration environment. Configure
``RAY_GRAVITINO_TEST_MAIN_URI``, ``RAY_GRAVITINO_TEST_METALAKE``,
``RAY_GRAVITINO_TEST_ICEBERG_REST_URI``, and the table identifiers documented
below. The service must expose a dynamic Iceberg REST catalog and a generic
lakehouse catalog for Delta. The Iceberg read table should contain data in
multiple files; the write table must be disposable and have a string ``id``
column. Delta tables must declare ``format=delta``, ``external=true``, and
``location``.

Pre-create these tables and set their fully qualified ``catalog.schema.table``
identifiers in ``RAY_GRAVITINO_TEST_ICEBERG_READ_TABLE``,
``RAY_GRAVITINO_TEST_ICEBERG_WRITE_TABLE``,
``RAY_GRAVITINO_TEST_DELTA_READ_TABLE``,
``RAY_GRAVITINO_TEST_DELTA_DENIED_TABLE``, and
``RAY_GRAVITINO_TEST_AUTH_DENIED_TABLE``. The Delta denied table must be an
existing external Delta table. The storage-denial case supplies invalid storage
options for that table. Set
``RAY_GRAVITINO_TEST_DENIED_STORAGE_OPTIONS`` to JSON containing invalid object
store credentials for the storage-denial case. Optionally set
``RAY_GRAVITINO_TEST_STORAGE_OPTIONS`` to JSON with the worker storage settings
used by Delta reads.

Authentication stays in environment variables. Set
``RAY_GRAVITINO_TEST_AUTH_MODE=simple`` and
``RAY_GRAVITINO_TEST_USERNAME`` for the main Gravitino API, or use ``basic``
with the corresponding username and password when the service is configured
for built-in IdP authentication. Provide REST-specific PyIceberg settings as JSON in
``RAY_GRAVITINO_TEST_ICEBERG_REST_KWARGS`` when needed. The denied-access cases
use ``RAY_GRAVITINO_TEST_DENIED_AUTH_MODE=simple`` with
``RAY_GRAVITINO_TEST_DENIED_USERNAME``, or ``basic`` with the denied username
and password, plus explicitly invalid storage options.
The test catalog uses a 30-second Gravitino client request timeout by default;
override it with JSON in ``RAY_GRAVITINO_TEST_CLIENT_CONFIG`` when needed.

This suite is registered as a Bazel ``manual`` target because it requires an
external Gravitino service and pre-created test metadata. It does not skip when
configuration is missing; an explicit invocation fails with the missing setting.
Install ``apache-gravitino``, ``pyiceberg``, and ``deltalake`` in the test virtual
environment before running it.
"""

import json
import os
import re
import uuid
from unittest import mock

import pytest

import ray
from ray.data.catalog import GravitinoCatalog, ReaderFormat

# conftest provides ray_start_regular_shared.
from ray.data.tests.conftest import *  # noqa: F401,F403


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"Set {name} to run the Gravitino service integration tests.")
    return value


def _auth_data_provider(prefix: str):
    mode = os.environ.get(f"{prefix}_AUTH_MODE", "none").lower()
    if mode == "none":
        return None

    if mode == "simple":
        try:
            from gravitino.auth.simple_auth_provider import SimpleAuthProvider
        except ImportError as e:
            pytest.fail(
                "Install the optional apache-gravitino Python client to run this "
                f"integration suite: {e}"
            )

        username = _required_env(f"{prefix}_USERNAME")
        with mock.patch.dict(os.environ, {"GRAVITINO_USER": username}):
            return SimpleAuthProvider()

    if mode != "basic":
        pytest.fail(f"{prefix}_AUTH_MODE must be 'none', 'simple', or 'basic'.")

    try:
        from gravitino.auth.basic_auth_provider import BasicAuthProvider
    except ImportError as e:
        pytest.fail(
            "Install the optional apache-gravitino Python client to run this "
            f"integration suite: {e}"
        )

    return BasicAuthProvider(
        _required_env(f"{prefix}_USERNAME"),
        _required_env(f"{prefix}_PASSWORD"),
    )


def _make_catalog(auth_prefix="RAY_GRAVITINO_TEST") -> GravitinoCatalog:
    rest_kwargs = json.loads(
        os.environ.get("RAY_GRAVITINO_TEST_ICEBERG_REST_KWARGS", "{}")
    )
    client_config = json.loads(
        os.environ.get(
            "RAY_GRAVITINO_TEST_CLIENT_CONFIG",
            '{"gravitino_client_request_timeout": 30}',
        )
    )
    return GravitinoCatalog(
        gravitino_uri=_required_env("RAY_GRAVITINO_TEST_MAIN_URI"),
        metalake_name=_required_env("RAY_GRAVITINO_TEST_METALAKE"),
        iceberg_rest_uri=_required_env("RAY_GRAVITINO_TEST_ICEBERG_REST_URI"),
        auth_data_provider=_auth_data_provider(auth_prefix),
        client_config=client_config,
        iceberg_rest_catalog_kwargs=rest_kwargs,
    )


def _test_storage_options():
    value = os.environ.get("RAY_GRAVITINO_TEST_STORAGE_OPTIONS")
    return json.loads(value) if value else None


@pytest.fixture(scope="module")
def gravitino_catalog():
    return _make_catalog()


def test_gravitino_iceberg_read_and_write(ray_start_regular_shared, gravitino_catalog):
    read_table = _required_env("RAY_GRAVITINO_TEST_ICEBERG_READ_TABLE")
    write_table = _required_env("RAY_GRAVITINO_TEST_ICEBERG_WRITE_TABLE")

    rows = ray.data.read_iceberg(
        table_identifier=read_table,
        catalog=gravitino_catalog,
        override_num_blocks=4,
    ).take_all()
    assert rows

    marker = f"ray-gravitino-{uuid.uuid4().hex}"
    ray.data.from_items([{"id": marker}]).write_iceberg(
        write_table, catalog=gravitino_catalog
    )
    written_rows = ray.data.read_iceberg(
        table_identifier=write_table,
        catalog=gravitino_catalog,
        override_num_blocks=4,
    ).take_all()
    assert marker in {row["id"] for row in written_rows}


def test_gravitino_external_delta_read_and_storage_denial(
    ray_start_regular_shared, gravitino_catalog
):
    table = _required_env("RAY_GRAVITINO_TEST_DELTA_READ_TABLE")
    rows = ray.data.read_delta(
        table,
        catalog=gravitino_catalog,
        storage_options=_test_storage_options(),
        override_num_blocks=4,
    ).take_all()
    assert rows

    denied_table = _required_env("RAY_GRAVITINO_TEST_DELTA_DENIED_TABLE")
    storage_options = json.loads(
        _required_env("RAY_GRAVITINO_TEST_DENIED_STORAGE_OPTIONS")
    )
    with pytest.raises(
        Exception,
        match=re.compile(
            r"access.?denied|forbidden|permission|403|InvalidAccessKeyId|"
            r"SignatureDoesNotMatch",
            re.IGNORECASE,
        ),
    ):
        ray.data.read_delta(
            denied_table,
            catalog=gravitino_catalog,
            storage_options=storage_options,
            override_num_blocks=4,
        ).take_all()


def test_gravitino_table_authorization_denial():
    denied_auth_mode = _required_env("RAY_GRAVITINO_TEST_DENIED_AUTH_MODE")
    if denied_auth_mode.lower() not in ("simple", "basic"):
        pytest.fail("RAY_GRAVITINO_TEST_DENIED_AUTH_MODE must be 'simple' or 'basic'.")

    try:
        from gravitino.exceptions.base import ForbiddenException
    except ImportError as e:
        pytest.fail(
            "Install the optional apache-gravitino Python client to run this "
            f"integration suite: {e}"
        )

    denied_catalog = _make_catalog("RAY_GRAVITINO_TEST_DENIED")
    denied_table = _required_env("RAY_GRAVITINO_TEST_AUTH_DENIED_TABLE")

    with pytest.raises(ForbiddenException):
        denied_catalog.resolve(denied_table, reader=ReaderFormat.DELTA)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
