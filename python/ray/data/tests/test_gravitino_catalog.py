"""Tests for the optional Gravitino Catalog connector."""

import builtins
import pickle
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

import ray
import ray.cloudpickle as cloudpickle
from ray.data.catalog import CatalogAccessMode, GravitinoCatalog, ReaderFormat

# Install Ray's shared test plugins and fixtures.
from ray.data.tests.conftest import *  # noqa: F401,F403


@pytest.fixture
def mock_gravitino(monkeypatch):
    table = mock.MagicMock()
    table.properties.return_value = {
        "format": "delta",
        "external": "true",
        "location": "s3://bucket/sales/orders",
    }

    table_catalog = mock.MagicMock()
    table_catalog.load_table.return_value = table

    catalog = mock.MagicMock()
    catalog.type.return_value = SimpleNamespace(type_name="relational")
    catalog.provider.return_value = "lakehouse-generic"
    catalog.as_table_catalog.return_value = table_catalog

    client = mock.MagicMock()
    client.load_catalog.return_value = catalog
    client_factory = mock.Mock(return_value=client)

    class FakeNameIdentifier:
        @staticmethod
        def of(*levels):
            return levels

    class FakePrivilege:
        class Name:
            SELECT_TABLE = "SELECT_TABLE"
            MODIFY_TABLE = "MODIFY_TABLE"

    gravitino_module = ModuleType("gravitino")
    gravitino_module.__path__ = []
    api_package = ModuleType("gravitino.api")
    api_package.__path__ = []
    authorization_package = ModuleType("gravitino.api.authorization")
    authorization_package.__path__ = []
    privilege_module = ModuleType("gravitino.api.authorization.privileges")
    privilege_module.Privilege = FakePrivilege
    client_package = ModuleType("gravitino.client")
    client_package.__path__ = []
    client_module = ModuleType("gravitino.client.gravitino_client")
    client_module.GravitinoClient = client_factory
    name_identifier_module = ModuleType("gravitino.name_identifier")
    name_identifier_module.NameIdentifier = FakeNameIdentifier

    for name, module in (
        ("gravitino", gravitino_module),
        ("gravitino.api", api_package),
        ("gravitino.api.authorization", authorization_package),
        ("gravitino.api.authorization.privileges", privilege_module),
        ("gravitino.client", client_package),
        ("gravitino.client.gravitino_client", client_module),
        ("gravitino.name_identifier", name_identifier_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        catalog=catalog,
        client=client,
        client_factory=client_factory,
        table=table,
        table_catalog=table_catalog,
        privilege_names=FakePrivilege.Name,
    )


@pytest.fixture
def gravitino_catalog():
    return GravitinoCatalog(
        gravitino_uri="http://gravitino:8090/",
        metalake_name="production",
        iceberg_rest_uri="http://gravitino:9001/iceberg",
    )


def test_gravitino_catalog_is_exported_from_ray_data():
    assert ray.data.GravitinoCatalog is GravitinoCatalog


@pytest.mark.parametrize("mode", [CatalogAccessMode.READ, CatalogAccessMode.WRITE])
def test_resolve_iceberg_catalog(gravitino_catalog, mock_gravitino, mode):
    mock_gravitino.catalog.provider.return_value = "lakehouse-iceberg"

    resolved = gravitino_catalog.resolve(
        "iceberg_catalog.sales.orders", reader=ReaderFormat.ICEBERG, mode=mode
    )

    assert resolved.table_identifier == "sales.orders"
    assert resolved.data_format is ReaderFormat.ICEBERG
    assert resolved.catalog_kwargs == {
        "type": "rest",
        "uri": "http://gravitino:9001/iceberg/",
        "warehouse": "iceberg_catalog",
        "header.X-Iceberg-Access-Delegation": "vended-credentials",
    }
    mock_gravitino.client.load_catalog.assert_called_once_with("iceberg_catalog")
    expected_privilege = (
        mock_gravitino.privilege_names.SELECT_TABLE
        if mode is CatalogAccessMode.READ
        else mock_gravitino.privilege_names.MODIFY_TABLE
    )
    mock_gravitino.table_catalog.load_table.assert_called_once_with(
        ("sales", "orders"), required_privilege_names={expected_privilege}
    )
    mock_gravitino.client.close.assert_not_called()


def test_resolve_iceberg_keeps_rest_auth_separate(gravitino_catalog, mock_gravitino):
    mock_gravitino.catalog.provider.return_value = "lakehouse-iceberg"
    main_api_auth = SimpleNamespace(username="metadata-user", password="main-secret")
    catalog = GravitinoCatalog(
        gravitino_uri="http://gravitino:8090",
        metalake_name="production",
        iceberg_rest_uri="https://iceberg:9001/iceberg/",
        auth_data_provider=main_api_auth,
        client_config={"gravitino_client_request_timeout": 30},
        iceberg_rest_catalog_kwargs={
            "credential": "rest-secret",
            "scope": "read",
            "type": "hive",
            "uri": "https://wrong-endpoint",
            "warehouse": "wrong-catalog",
        },
    )

    resolved = catalog.resolve(
        "iceberg_catalog.sales.orders", reader=ReaderFormat.ICEBERG
    )

    assert resolved.catalog_kwargs["uri"] == "https://iceberg:9001/iceberg/"
    assert resolved.catalog_kwargs["warehouse"] == "iceberg_catalog"
    assert resolved.catalog_kwargs["credential"] == "rest-secret"
    assert resolved.catalog_kwargs["type"] == "rest"
    mock_gravitino.client_factory.assert_called_once_with(
        uri="http://gravitino:8090",
        metalake_name="production",
        auth_data_provider=main_api_auth,
        client_config={"gravitino_client_request_timeout": 30},
    )


def test_resolve_external_delta_table(gravitino_catalog, mock_gravitino):
    resolved = gravitino_catalog.resolve(
        "lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA
    )

    assert resolved.path == "s3://bucket/sales/orders"
    assert resolved.data_format is ReaderFormat.DELTA
    assert resolved.catalog_kwargs is None
    mock_gravitino.table_catalog.load_table.assert_called_once_with(
        ("sales", "orders"),
        required_privilege_names={mock_gravitino.privilege_names.SELECT_TABLE},
    )


def test_repeated_resolve_keeps_caller_auth_provider_open(mock_gravitino):
    auth_provider = SimpleNamespace(username="metadata-user")
    catalog = GravitinoCatalog(
        gravitino_uri="http://gravitino:8090",
        metalake_name="production",
        iceberg_rest_uri="http://gravitino:9001/iceberg/",
        auth_data_provider=auth_provider,
    )

    catalog.resolve("lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA)
    catalog.resolve("lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA)

    assert mock_gravitino.client_factory.call_count == 2
    assert all(
        call.kwargs["auth_data_provider"] is auth_provider
        for call in mock_gravitino.client_factory.call_args_list
    )
    mock_gravitino.client.close.assert_not_called()


def test_read_iceberg_uses_gravitino_resolution(gravitino_catalog, mock_gravitino):
    mock_gravitino.catalog.provider.return_value = "lakehouse-iceberg"
    dataset = object()

    with (
        mock.patch(
            "ray.data._internal.datasource.iceberg_datasource.IcebergDatasource"
        ) as datasource,
        mock.patch("ray.data.read_api.read_datasource", return_value=dataset),
    ):
        result = ray.data.read_iceberg(
            table_identifier="iceberg_catalog.sales.orders",
            catalog=gravitino_catalog,
        )

    assert result is dataset
    assert datasource.call_args.kwargs["table_identifier"] == "sales.orders"
    assert datasource.call_args.kwargs["catalog_kwargs"]["warehouse"] == (
        "iceberg_catalog"
    )


def test_read_delta_uses_gravitino_table_location(gravitino_catalog, mock_gravitino):
    delta_calls = {}
    fake_delta_module = ModuleType("deltalake")

    class FakeDeltaTable:
        def __init__(self, path, version=None, storage_options=None):
            delta_calls["path"] = path
            delta_calls["version"] = version
            delta_calls["storage_options"] = storage_options

        def to_pyarrow_dataset(self, filesystem=None):
            delta_calls["filesystem"] = filesystem
            return object()

    fake_delta_module.DeltaTable = FakeDeltaTable

    with (
        mock.patch.dict(sys.modules, {"deltalake": fake_delta_module}),
        mock.patch(
            "ray.data._internal.datasource.parquet_datasource.ParquetDatasource.from_pyarrow_dataset",
            return_value=object(),
        ),
        mock.patch("ray.data.read_api.read_datasource", return_value=object()) as read,
    ):
        result = ray.data.read_delta(
            "lakehouse_catalog.sales.orders", catalog=gravitino_catalog
        )

    assert delta_calls["path"] == "s3://bucket/sales/orders"
    assert read.called
    assert result is not None


def test_write_iceberg_uses_gravitino_resolution(gravitino_catalog, mock_gravitino):
    mock_gravitino.catalog.provider.return_value = "lakehouse-iceberg"
    dataset = object.__new__(ray.data.Dataset)

    with (
        mock.patch("ray.data.dataset.IcebergDatasink") as datasink,
        mock.patch.object(ray.data.Dataset, "write_datasink"),
    ):
        ray.data.Dataset.write_iceberg(
            dataset, "iceberg_catalog.sales.orders", catalog=gravitino_catalog
        )

    assert datasink.call_args.kwargs["table_identifier"] == "sales.orders"
    assert datasink.call_args.kwargs["catalog_kwargs"]["warehouse"] == (
        "iceberg_catalog"
    )


@pytest.mark.parametrize(
    "provider,properties,error",
    [
        (
            "lakehouse-iceberg",
            {"format": "delta", "external": "true", "location": "s3://bucket/t"},
            "lakehouse-generic",
        ),
        (
            "lakehouse-generic",
            {"format": "iceberg", "external": "true", "location": "s3://bucket/t"},
            "format='delta'",
        ),
        (
            "lakehouse-generic",
            {"format": "delta", "external": "false", "location": "s3://bucket/t"},
            "external='true'",
        ),
        (
            "lakehouse-generic",
            {"format": "delta", "external": "true"},
            "table-level location",
        ),
    ],
)
def test_resolve_delta_validates_registered_metadata(
    gravitino_catalog, mock_gravitino, provider, properties, error
):
    mock_gravitino.catalog.provider.return_value = provider
    mock_gravitino.table.properties.return_value = properties

    with pytest.raises(ValueError, match=error):
        gravitino_catalog.resolve(
            "lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA
        )


def test_resolve_rejects_delta_write_before_contacting_gravitino(
    gravitino_catalog, mock_gravitino
):
    with pytest.raises(ValueError, match="Delta writes through GravitinoCatalog"):
        gravitino_catalog.resolve(
            "lakehouse_catalog.sales.orders",
            reader=ReaderFormat.DELTA,
            mode=CatalogAccessMode.WRITE,
        )

    mock_gravitino.client_factory.assert_not_called()


def test_resolve_rejects_parquet_before_contacting_gravitino(
    gravitino_catalog, mock_gravitino
):
    with pytest.raises(ValueError, match="does not support format='parquet'"):
        gravitino_catalog.resolve(
            "hive_catalog.sales.orders", reader=ReaderFormat.PARQUET
        )

    mock_gravitino.client_factory.assert_not_called()


def test_resolve_rejects_non_relational_catalog(gravitino_catalog, mock_gravitino):
    mock_gravitino.catalog.type.return_value = SimpleNamespace(type_name="fileset")

    with pytest.raises(ValueError, match="relational Gravitino catalogs"):
        gravitino_catalog.resolve(
            "lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA
        )

    mock_gravitino.client.close.assert_not_called()


def test_resolve_propagates_gravitino_authorization_denial(
    gravitino_catalog, mock_gravitino
):
    mock_gravitino.table_catalog.load_table.side_effect = PermissionError("denied")

    with pytest.raises(PermissionError, match="denied"):
        gravitino_catalog.resolve(
            "lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA
        )

    mock_gravitino.table_catalog.load_table.assert_called_once_with(
        ("sales", "orders"),
        required_privilege_names={mock_gravitino.privilege_names.SELECT_TABLE},
    )
    mock_gravitino.client.close.assert_not_called()


def test_resolve_rejects_invalid_reader_before_contacting_gravitino(
    gravitino_catalog, mock_gravitino
):
    with pytest.raises(ValueError, match="reader must be"):
        gravitino_catalog.resolve("lakehouse_catalog.sales.orders", reader=None)

    mock_gravitino.client_factory.assert_not_called()


@pytest.mark.parametrize(
    "identifier", ["orders", "sales.orders", "catalog.sales.orders.extra", "a..c"]
)
def test_resolve_requires_three_part_table_identifier(
    gravitino_catalog, mock_gravitino, identifier
):
    with pytest.raises(ValueError, match="three-part catalog.schema.table"):
        gravitino_catalog.resolve(identifier, reader=ReaderFormat.DELTA)

    mock_gravitino.client_factory.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gravitino_uri": "http://user:secret@gravitino:8090"},
        {"iceberg_rest_uri": "http://iceberg:9001/iceberg?token=secret"},
        {"gravitino_uri": "relative/path"},
    ],
)
def test_init_rejects_embedded_credentials_or_invalid_endpoint(kwargs):
    config = {
        "gravitino_uri": "http://gravitino:8090",
        "metalake_name": "production",
        "iceberg_rest_uri": "http://iceberg:9001/iceberg",
    }
    config.update(kwargs)

    with pytest.raises(ValueError):
        GravitinoCatalog(**config)


@pytest.mark.parametrize(
    "client_config",
    [
        {"access_token": "client-config-secret"},
        {"gravitino_client_request_timeout": "client-config-secret"},
        {"gravitino_client_request_timeout": True},
        {"gravitino_client_request_timeout": -1},
    ],
)
def test_init_rejects_unsupported_or_sensitive_client_config(client_config):
    with pytest.raises(ValueError, match="client_config") as exc_info:
        GravitinoCatalog(
            gravitino_uri="http://gravitino:8090",
            metalake_name="production",
            iceberg_rest_uri="http://iceberg:9001/iceberg",
            client_config=client_config,
        )

    assert "client-config-secret" not in str(exc_info.value)


def test_optional_gravitino_client_has_actionable_import_error(
    gravitino_catalog, monkeypatch
):
    original_import = builtins.__import__

    def import_without_gravitino(name, *args, **kwargs):
        if name.startswith("gravitino"):
            raise ModuleNotFoundError("No module named 'gravitino'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gravitino)

    with pytest.raises(ImportError, match="pip install apache-gravitino"):
        gravitino_catalog.resolve(
            "lakehouse_catalog.sales.orders", reader=ReaderFormat.DELTA
        )


def test_catalog_repr_and_pickle_omit_credentials():
    main_secret = "main-api-secret-6cdb"
    rest_secret = "iceberg-rest-secret-9bcf"
    catalog = GravitinoCatalog(
        gravitino_uri="http://gravitino:8090",
        metalake_name="production",
        iceberg_rest_uri="http://iceberg:9001/iceberg",
        auth_data_provider=SimpleNamespace(password=main_secret),
        client_config={"gravitino_client_request_timeout": 30},
        iceberg_rest_catalog_kwargs={"credential": rest_secret},
    )

    representation = repr(catalog)
    assert main_secret not in representation
    assert rest_secret not in representation

    for serializer in (pickle, cloudpickle):
        serialized = serializer.dumps(catalog)
        assert main_secret.encode() not in serialized
        assert rest_secret.encode() not in serialized

        restored = serializer.loads(serialized)
        assert restored.client_config == {"gravitino_client_request_timeout": 30}
        assert restored._credentials_redacted
        with pytest.raises(
            RuntimeError, match="deserialized without its authentication"
        ):
            restored.resolve(
                "iceberg_catalog.sales.orders", reader=ReaderFormat.ICEBERG
            )
