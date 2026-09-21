"""Catalog connectors for Ray Data readers.

A :class:`Catalog` resolves a table name into a readable source (location +
credentials) for a reader such as :func:`ray.data.read_delta`,
:func:`ray.data.read_parquet`, or :func:`ray.data.read_iceberg`.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from urllib.parse import urljoin, urlsplit

from ray.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    import pyarrow.fs
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        AwsCredentials,
        AzureUserDelegationSas,
        GcpOauthToken,
        GenerateTemporaryTableCredentialResponse,
        TableInfo,
        TableOperation,
    )
    from gravitino.auth.auth_data_provider import AuthDataProvider

    from ray.data._internal.datasource.databricks_credentials import (
        DatabricksCredentialProvider,
    )

logger = logging.getLogger(__name__)

_DELTA_UNIFORM_FORMATS_PROPERTY = "delta.universalFormat.enabledFormats"

# Environment-variable names the underlying readers (pyarrow / deltalake's
# object_store) pick up vended credentials from.
_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
_AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
_AWS_REGION = "AWS_REGION"
_AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
_AZURE_STORAGE_SAS_TOKEN = "AZURE_STORAGE_SAS_TOKEN"


def _normalize_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return host


@PublicAPI(stability="alpha")
class ReaderFormat(str, Enum):
    """Which reader is asking the catalog to resolve a table."""

    DELTA = "delta"
    PARQUET = "parquet"
    ICEBERG = "iceberg"


@DeveloperAPI
class CatalogAccessMode(str, Enum):
    """Whether the catalog should vend read or write credentials."""

    READ = "read"
    WRITE = "write"

    def as_databricks_table_op(self) -> "TableOperation":
        # Unity Catalog only exposes READ and READ_WRITE (there is no write-only
        # operation), so WRITE maps to READ_WRITE.
        from databricks.sdk.service.catalog import TableOperation

        if self == CatalogAccessMode.READ:
            return TableOperation.READ
        elif self == CatalogAccessMode.WRITE:
            return TableOperation.READ_WRITE
        raise ValueError("Unsupported CatalogAccessMode for Databricks TableOperation")


@DeveloperAPI
@dataclass
class ResolvedSource:
    """The output of :meth:`Catalog.resolve` — location/credentials for a reader.

    A reader consumes only the fields it understands:

    * ``read_delta``:   ``path`` + (``storage_options`` and/or ``filesystem``)
    * ``read_parquet``: ``path`` + ``filesystem``
    * ``read_iceberg``: ``catalog_kwargs`` + ``table_identifier``

    Unused fields are ``None``.
    """

    path: Optional[str] = None
    filesystem: Optional["pyarrow.fs.FileSystem"] = None
    storage_options: Optional[Dict[str, Any]] = None
    catalog_kwargs: Optional[Dict[str, Any]] = None
    # Identifier the reader should address the table by, if the catalog rewrites
    # it (e.g. Iceberg REST scopes the warehouse to the catalog, so the table is
    # addressed as ``schema.table`` rather than ``catalog.schema.table``).
    table_identifier: Optional[str] = None
    data_format: Optional[ReaderFormat] = None  # hint, e.g. ReaderFormat.DELTA


@PublicAPI(stability="alpha")
class Catalog(ABC):
    """A directory service that resolves a table name to a readable source."""

    @abstractmethod
    def resolve(
        self,
        table: str,
        *,
        reader: ReaderFormat,
        mode: CatalogAccessMode = CatalogAccessMode.READ,
    ) -> ResolvedSource:
        """Resolve ``table`` for the given ``reader`` and access ``mode``."""
        ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class DatabricksUnityCatalog(Catalog):
    """Databricks Unity Catalog connector.

    For Delta and Parquet tables this performs Unity Catalog credential vending
    (temporary, least-privilege cloud credentials). For Iceberg tables it
    returns configuration pointing PyIceberg at Unity Catalog's Iceberg REST
    catalog endpoint.

    Args:
        url: Databricks workspace URL (e.g.
            ``"https://dbc-XXXX.cloud.databricks.com"``). Required unless
            ``credential_provider`` is given.
        token: Databricks Personal Access Token with ``EXTERNAL USE SCHEMA``
            permission. Required unless ``credential_provider`` is given.
        credential_provider: A custom
            :class:`~ray.data._internal.datasource.databricks_credentials.DatabricksCredentialProvider`.
            If provided, ``url``/``token`` are ignored.
        region: AWS region for S3 access (e.g. ``"us-west-2"``). Required for
            AWS-backed tables; not needed for Azure/GCP.

    Example:
        >>> import ray
        >>> catalog = ray.data.DatabricksUnityCatalog(  # doctest: +SKIP
        ...     url="https://dbc-XXXX.cloud.databricks.com",
        ...     token="dapi...",
        ...     region="us-west-2",
        ... )
        >>> ds = ray.data.read_delta(  # doctest: +SKIP
        ...     "main.sales.transactions", catalog=catalog
        ... )
    """

    _: KW_ONLY
    url: Optional[str] = None
    # `repr=False` keeps the token/provider out of the auto-generated repr.
    token: Optional[str] = field(default=None, repr=False)
    credential_provider: Optional["DatabricksCredentialProvider"] = field(
        default=None, repr=False
    )
    region: Optional[str] = None

    # Derived in __post_init__; declared (init=False) so type checkers know the
    # attributes exist, and excluded from repr/eq.
    _provider: "DatabricksCredentialProvider" = field(
        init=False, repr=False, compare=False
    )
    _base_url: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from ray.data._internal.datasource.databricks_credentials import (
            UnityCatalogCredentialConfig,
            resolve_credential_provider,
        )

        # Derived (not init args); `object.__setattr__` is how a frozen dataclass
        # assigns inside __post_init__.
        provider = resolve_credential_provider(
            UnityCatalogCredentialConfig(
                credential_provider=self.credential_provider,
                url=self.url,
                token=self.token,
            )
        )
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_base_url", _normalize_host(provider.get_host()))

    # ---- Catalog interface -------------------------------------------------
    def resolve(
        self,
        table: str,
        *,
        reader: ReaderFormat,
        mode: CatalogAccessMode = CatalogAccessMode.READ,
    ) -> ResolvedSource:
        assert reader is not None and isinstance(reader, ReaderFormat)
        assert mode is not None and isinstance(mode, CatalogAccessMode)
        if reader is ReaderFormat.ICEBERG:
            return self._resolve_iceberg(table)
        if reader in (ReaderFormat.DELTA, ReaderFormat.PARQUET):
            return self._resolve_storage(table, reader, mode)
        # Reached only if a new ReaderFormat is added without handling here.
        raise ValueError(f"DatabricksUnityCatalog does not support format={reader!r}")

    # ---- storage-credential vending (delta / parquet) ----------------------
    def _resolve_storage(
        self, table: str, reader: ReaderFormat, mode: CatalogAccessMode
    ) -> ResolvedSource:
        table_info = self._get_table_info(table)
        creds, table_url = self._get_creds(table_info.table_id, mode)

        # Some readers/writers need an explicit pyarrow filesystem:
        #  - AWS Delta: the vended session token isn't reliably propagated through
        #    `DeltaTable.to_pyarrow_dataset`'s auto-built filesystem.
        #  - AWS write: an S3FileSystem built from the default credential chain
        #    (the `filesystem=None` path) does NOT serialize its credentials, so a
        #    worker would rebuild it from *its own* environment. Reads get away
        #    with this because `_apply_env` can seed the vended creds into the
        #    cluster `runtime_env` while Ray is still uninitialized; a write always
        #    runs after Ray is initialized (a materialized Dataset already exists),
        #    so that propagation is unavailable. Build an explicit S3FileSystem
        #    whose creds *do* pickle into the datasink and reach the workers.
        #  - GCP Parquet: a bare OAuth token has no env var pyarrow auto-reads,
        #    so the data scan needs an explicit GcsFileSystem.
        filesystem = None
        if creds.aws_temp_credentials is not None and (
            reader is ReaderFormat.DELTA or mode is CatalogAccessMode.WRITE
        ):
            filesystem = self._build_s3_filesystem(creds.aws_temp_credentials)
        elif creds.gcp_oauth_token is not None:
            if reader is ReaderFormat.DELTA:
                # Unity Catalog vends a GCP OAuth token, but deltalake's
                # object_store (<=0.13.x, bundled in deltalake<=1.6.1) only
                # accepts service-account-key auth for GCS -- it has no
                # bearer/OAuth-token config key -- so the Delta transaction-log
                # read can't use the vended token and silently falls back to GCE
                # metadata-server auth.
                raise RuntimeError(
                    "Reading a GCP-backed Delta table via Unity Catalog "
                    "credential vending is not supported as deltalake "
                    "does not have the required object_store version."
                )
            filesystem = self._build_gcs_filesystem(
                creds.gcp_oauth_token, creds.expiration_time
            )

        # Deliver vended credentials via environment variables. This is the
        # mechanism the underlying libraries read uniformly: pyarrow (Parquet
        # data, and S3/Azure/GCS auto-filesystems) and deltalake's object_store
        # (the Delta transaction *log* read in `DeltaTable(...)`, which neither
        # a pyarrow `filesystem` nor `storage_options` keyed for pyarrow would
        # satisfy). See `_apply_env` for the worker-propagation note.
        #
        # TODO: remove the env-var + ray.init mechanism once credential vending
        # is performed inside the read tasks themselves (worker-side).
        self._apply_env(self._creds_to_env(creds))

        return ResolvedSource(
            path=table_url,
            filesystem=filesystem,
            data_format=self._infer_format(table_info, table_url),
        )

    # ---- iceberg REST catalog ---------------------------------------------
    def _resolve_iceberg(self, table: str) -> ResolvedSource:
        # PyIceberg speaks the Iceberg REST protocol; Unity Catalog implements
        # it and vends data-file credentials via the access-delegation header.
        # No manual S3/ADLS/GCS keys are needed here.
        #
        # The REST catalog is scoped to a single UC catalog via `warehouse`, so
        # the table is addressed by `schema.table` (the catalog prefix would
        # otherwise be double-applied, e.g. `tmp.tmp.schema.table`).
        catalog_name, _, namespace_table = table.partition(".")
        return ResolvedSource(
            table_identifier=namespace_table,
            catalog_kwargs={
                "type": "rest",
                "uri": urljoin(self._base_url, "/api/2.1/unity-catalog/iceberg-rest"),
                "warehouse": catalog_name,
                "token": self._provider.get_token(),
                "header.X-Iceberg-Access-Delegation": "vended-credentials",
            },
            data_format=ReaderFormat.ICEBERG,
        )

    # ---- Unity Catalog SDK helpers ----------------------------------------
    def _workspace_client(self) -> "WorkspaceClient":
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient(host=self._base_url, token=self._provider.get_token())

    def _call_with_token_refresh(self, call: Callable) -> Any:
        """Run ``call(workspace_client)``, retrying once on 401.

        Mirrors the previous ``request_with_401_retry`` behavior: on an
        authentication failure, invalidate the credential provider (so the next
        ``get_token()`` returns a fresh token) and retry once with a new client.
        Matters for refreshable providers; a no-op for static PATs.
        """
        from databricks.sdk.errors import Unauthenticated

        try:
            return call(self._workspace_client())
        except Unauthenticated:
            logger.info("Received 401 from Unity Catalog; refreshing credentials.")
            self._provider.invalidate()
            return call(self._workspace_client())

    def _get_table_info(self, table: str) -> "TableInfo":
        return self._call_with_token_refresh(lambda w: w.tables.get(full_name=table))

    def _get_creds(
        self, table_id: Optional[str], mode: CatalogAccessMode = CatalogAccessMode.READ
    ) -> Tuple["GenerateTemporaryTableCredentialResponse", str]:
        assert table_id is not None
        operation = mode.as_databricks_table_op()
        creds = self._call_with_token_refresh(
            lambda w: w.temporary_table_credentials.generate_temporary_table_credentials(
                table_id=table_id, operation=operation
            )
        )
        return creds, creds.url

    @staticmethod
    def _infer_format(
        table_info: "TableInfo", table_url: str
    ) -> Optional[ReaderFormat]:
        """Best-effort format hint from table metadata or file extension."""
        from databricks.sdk.service.catalog import DataSourceFormat

        dsf = table_info.data_source_format
        if dsf == DataSourceFormat.DELTA:
            uniform = (table_info.properties or {}).get(
                _DELTA_UNIFORM_FORMATS_PROPERTY, ""
            )
            if "iceberg" in uniform.lower():
                return ReaderFormat.ICEBERG
            return ReaderFormat.DELTA
        elif dsf == DataSourceFormat.PARQUET:
            return ReaderFormat.PARQUET

        storage_loc = table_info.storage_location or table_url
        if storage_loc:
            ext = os.path.splitext(storage_loc)[-1].replace(".", "").lower()
            if ext in (ReaderFormat.DELTA.value, ReaderFormat.PARQUET.value):
                return ReaderFormat(ext)
        return None

    def infer_format(self, table: str) -> Optional[ReaderFormat]:
        """Best-effort format hint from table metadata or file extension.

        Calling this function will query DatabricksUnityCatalog to get the
        relevant information."""
        info = self._get_table_info(table)
        _, table_url = self._get_creds(info.table_id)
        return self._infer_format(info, table_url)

    def _creds_to_env(
        self, creds: "GenerateTemporaryTableCredentialResponse"
    ) -> Dict[str, Optional[str]]:
        """Translate vended credentials into environment variables."""
        if creds.aws_temp_credentials is not None:
            aws = creds.aws_temp_credentials
            env = {
                _AWS_ACCESS_KEY_ID: aws.access_key_id,
                _AWS_SECRET_ACCESS_KEY: aws.secret_access_key,
                _AWS_SESSION_TOKEN: aws.session_token,
            }
            if self.region:
                env[_AWS_REGION] = self.region
                env[_AWS_DEFAULT_REGION] = self.region
            return env

        if creds.azure_user_delegation_sas is not None:
            return self._parse_azure_creds(creds.azure_user_delegation_sas)

        if creds.gcp_oauth_token is not None:
            # A bare GCP OAuth token has no env var pyarrow/deltalake auto-read;
            # it's delivered via an explicit GcsFileSystem (data scan) and via
            # `storage_options` (Delta log read) in `_resolve_storage` instead.
            return {}

        raise ValueError("No known credential type found in Databricks UC response.")

    @staticmethod
    def _apply_env(env_vars: Dict[str, Optional[str]]) -> None:
        """Set vended credentials in the environment and propagate to workers.

        Credentials are set on the driver's ``os.environ`` and, if Ray has not
        been initialized yet, into the cluster ``runtime_env`` so read tasks on
        workers inherit them. If Ray is already running we cannot retroactively
        amend its ``runtime_env``; driver-side env still covers driver reads
        (e.g. the Delta log) and single-node execution.

        TODO: remove once credential vending happens inside the read tasks.
        """
        import ray

        if not env_vars:
            return

        for k, v in env_vars.items():
            if v:
                os.environ[k] = v
        if not ray.is_initialized():
            ray.init(runtime_env={"env_vars": dict(env_vars)})

    def _build_s3_filesystem(self, aws: "AwsCredentials") -> "pyarrow.fs.FileSystem":
        if not self.region:
            raise ValueError(
                "The 'region' parameter is required for AWS S3 access. "
                "Please specify the AWS region (e.g., region='us-west-2')."
            )
        import pyarrow.fs as pafs

        return pafs.S3FileSystem(
            access_key=aws.access_key_id,
            secret_key=aws.secret_access_key,
            session_token=aws.session_token,
            region=self.region,
        )

    @staticmethod
    def _build_gcs_filesystem(
        gcp: "GcpOauthToken", expiration_time: Optional[int]
    ) -> "pyarrow.fs.FileSystem":
        import pyarrow.fs as pafs

        if expiration_time is None:
            # pyarrow requires an expiration alongside an access token.
            raise ValueError(
                "GCP credential vending did not return an expiration_time."
            )
        expiration = datetime.fromtimestamp(expiration_time / 1000, tz=timezone.utc)
        return pafs.GcsFileSystem(
            access_token=gcp.oauth_token,
            credential_token_expiration=expiration,
        )

    @staticmethod
    def _parse_azure_creds(sas: "AzureUserDelegationSas") -> Dict[str, Optional[str]]:
        sas_token = sas.sas_token
        if sas_token and sas_token.startswith("?"):
            sas_token = sas_token[1:]
        if not sas_token:
            raise ValueError("Azure UC credentials missing a SAS token.")
        creds: Dict[str, Optional[str]] = {_AZURE_STORAGE_SAS_TOKEN: sas_token}
        return creds


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GravitinoCatalog(Catalog):
    """Resolve Iceberg and registered external Delta tables.

    The Gravitino main API resolves the logical catalog and table metadata. Iceberg
    operations then use the separate Gravitino Iceberg REST endpoint through
    PyIceberg. Delta reads use the table's registered ``location`` and the
    storage credentials already configured for Ray workers.

    The optional ``apache-gravitino`` client is imported only when ``resolve`` is
    called. Authentication for the main API and Iceberg REST service is configured
    independently.

    Args:
        gravitino_uri: Base URI of the Gravitino main API.
        metalake_name: Metalake containing the catalogs to resolve.
        iceberg_rest_uri: Gravitino Iceberg REST endpoint, for example
            ``"http://gravitino:9001/iceberg/"``.
        auth_data_provider: Optional Gravitino Python client authentication
            provider. This authenticates requests to the main Gravitino API.
        client_config: Optional Gravitino 1.3 client configuration. It currently
            supports only ``gravitino_client_request_timeout``, a non-negative
            integer number of seconds.
        iceberg_rest_catalog_kwargs: Optional PyIceberg REST catalog properties,
            such as OAuth credentials for the Iceberg REST endpoint.

    Example:
        >>> import ray
        >>> catalog = ray.data.GravitinoCatalog(  # doctest: +SKIP
        ...     gravitino_uri="http://gravitino:8090",
        ...     metalake_name="production",
        ...     iceberg_rest_uri="http://gravitino:9001/iceberg/",
        ... )
        >>> ds = ray.data.read_iceberg(  # doctest: +SKIP
        ...     table_identifier="iceberg_catalog.sales.orders", catalog=catalog
        ... )
        >>> delta = ray.data.read_delta(  # doctest: +SKIP
        ...     "lakehouse_catalog.sales.orders", catalog=catalog
        ... )

    Note:
        Gravitino 1.3.0 supports external Delta table metadata, but does not vend
        storage credentials for Generic Lakehouse Delta tables. Configure the
        storage identity on the Ray workers. Delta writes are unsupported because
        Gravitino 1.3.0 does not support ALTER for external Delta tables, so table
        metadata consistency cannot be guaranteed after a write.
        Parquet access through this catalog is unsupported.
        Authentication configuration is omitted when the Catalog is pickled;
        recreate it before resolving from a deserialized instance.
    """

    _: KW_ONLY
    gravitino_uri: str
    metalake_name: str
    iceberg_rest_uri: str
    # These fields may contain credentials. Keep them out of repr and equality.
    auth_data_provider: Optional["AuthDataProvider"] = field(
        default=None, repr=False, compare=False
    )
    client_config: Optional[Dict[str, int]] = field(
        default=None, repr=False, compare=False
    )
    iceberg_rest_catalog_kwargs: Optional[Dict[str, Any]] = field(
        default=None, repr=False, compare=False
    )
    _credentials_redacted: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if not isinstance(self.metalake_name, str) or not self.metalake_name.strip():
            raise ValueError("metalake_name must be a non-empty string.")
        object.__setattr__(self, "metalake_name", self.metalake_name.strip())

        for name, uri in (
            ("gravitino_uri", self.gravitino_uri),
            ("iceberg_rest_uri", self.iceberg_rest_uri),
        ):
            if not isinstance(uri, str) or not uri.strip():
                raise ValueError(f"{name} must be a non-empty absolute HTTP(S) URI.")
            parsed_uri = urlsplit(uri.strip())
            if parsed_uri.scheme not in ("http", "https") or not parsed_uri.netloc:
                raise ValueError(f"{name} must be a non-empty absolute HTTP(S) URI.")
            if (
                parsed_uri.username is not None
                or parsed_uri.password is not None
                or parsed_uri.query
                or parsed_uri.fragment
            ):
                raise ValueError(
                    f"{name} cannot embed credentials, query parameters, or fragments; "
                    "configure authentication separately."
                )

        object.__setattr__(
            self, "gravitino_uri", self.gravitino_uri.strip().rstrip("/")
        )
        object.__setattr__(
            self, "iceberg_rest_uri", self.iceberg_rest_uri.strip().rstrip("/") + "/"
        )
        client_config = dict(self.client_config or {})
        timeout_key = "gravitino_client_request_timeout"
        if any(key != timeout_key for key in client_config):
            raise ValueError(
                "client_config only supports 'gravitino_client_request_timeout'."
            )
        if timeout_key in client_config:
            timeout = client_config[timeout_key]
            if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0:
                raise ValueError(
                    "client_config['gravitino_client_request_timeout'] must be "
                    "a non-negative integer."
                )
        object.__setattr__(self, "client_config", client_config)
        object.__setattr__(
            self,
            "iceberg_rest_catalog_kwargs",
            dict(self.iceberg_rest_catalog_kwargs or {}),
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Omit authentication material from a serialized Catalog instance.

        Ray resolves this catalog on the driver before creating read tasks or an
        Iceberg datasink. Those workers receive only the resulting ``ResolvedSource``
        or PyIceberg catalog properties. A deserialized ``GravitinoCatalog`` must
        be reconstructed with its authentication configuration before resolving.
        """
        state = self.__dict__.copy()
        has_auth_config = self.auth_data_provider is not None or bool(
            self.iceberg_rest_catalog_kwargs
        )
        state["auth_data_provider"] = None
        state["iceberg_rest_catalog_kwargs"] = {}
        state["_credentials_redacted"] = self._credentials_redacted or has_auth_config
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def _gravitino_client(self) -> Any:
        if self._credentials_redacted:
            raise RuntimeError(
                "This GravitinoCatalog was deserialized without its authentication "
                "configuration. Recreate it before calling resolve()."
            )

        try:
            from gravitino.client.gravitino_client import GravitinoClient
        except ImportError as e:
            raise ImportError(
                "GravitinoCatalog.resolve() requires the optional 'apache-gravitino' "
                "package. Install it with `pip install apache-gravitino`."
            ) from e

        return GravitinoClient(
            uri=self.gravitino_uri,
            metalake_name=self.metalake_name,
            auth_data_provider=self.auth_data_provider,
            client_config=self.client_config or None,
        )

    def _load_table_metadata(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        mode: CatalogAccessMode,
    ) -> tuple[str, Dict[str, str]]:
        if self._credentials_redacted:
            raise RuntimeError(
                "This GravitinoCatalog was deserialized without its authentication "
                "configuration. Recreate it before calling resolve()."
            )

        try:
            from gravitino.api.authorization.privileges import Privilege
            from gravitino.name_identifier import NameIdentifier
        except ImportError as e:
            raise ImportError(
                "GravitinoCatalog.resolve() requires the optional 'apache-gravitino' "
                "package. Install it with `pip install apache-gravitino`."
            ) from e

        client = self._gravitino_client()
        catalog = client.load_catalog(catalog_name)
        catalog_type = catalog.type()
        if getattr(catalog_type, "type_name", None) != "relational":
            raise ValueError(
                "GravitinoCatalog supports relational Gravitino catalogs only."
            )

        required_privilege = (
            Privilege.Name.SELECT_TABLE
            if mode is CatalogAccessMode.READ
            else Privilege.Name.MODIFY_TABLE
        )
        table = catalog.as_table_catalog().load_table(
            NameIdentifier.of(schema_name, table_name),
            required_privilege_names={required_privilege},
        )
        # Gravitino 1.3's HTTPClient creates an opener per request; its close()
        # sends an HTTP CLOSE request and closes the caller-owned auth provider.
        # Keep that provider reusable for subsequent Catalog.resolve() calls.
        return catalog.provider(), dict(table.properties() or {})

    @staticmethod
    def _split_table_identifier(table: str) -> tuple[str, str, str]:
        if not isinstance(table, str):
            raise ValueError("table must be a three-part catalog.schema.table string.")
        parts = table.split(".")
        if len(parts) != 3 or any(not part.strip() for part in parts):
            raise ValueError("table must be a three-part catalog.schema.table string.")
        return parts[0], parts[1], parts[2]

    def resolve(
        self,
        table: str,
        *,
        reader: ReaderFormat,
        mode: CatalogAccessMode = CatalogAccessMode.READ,
    ) -> ResolvedSource:
        if not isinstance(reader, ReaderFormat):
            raise ValueError("reader must be a ReaderFormat value.")
        if not isinstance(mode, CatalogAccessMode):
            raise ValueError("mode must be a CatalogAccessMode value.")
        if reader not in (
            ReaderFormat.ICEBERG,
            ReaderFormat.DELTA,
        ):
            raise ValueError(
                f"GravitinoCatalog does not support format={reader.value!r}."
            )

        if reader is ReaderFormat.DELTA and mode is CatalogAccessMode.WRITE:
            raise ValueError(
                "Delta writes through GravitinoCatalog are not supported because "
                "Gravitino 1.3.0 does not support ALTER for external Delta tables, "
                "so table metadata consistency cannot be guaranteed after a write."
            )

        catalog_name, schema_name, table_name = self._split_table_identifier(table)

        provider, properties = self._load_table_metadata(
            catalog_name, schema_name, table_name, mode
        )

        if reader is ReaderFormat.ICEBERG:
            if provider != "lakehouse-iceberg":
                raise ValueError(
                    "Iceberg access requires a Gravitino 'lakehouse-iceberg' catalog."
                )

            catalog_kwargs = dict(self.iceberg_rest_catalog_kwargs or {})
            catalog_kwargs.update(
                {
                    "type": "rest",
                    "uri": self.iceberg_rest_uri,
                    "warehouse": catalog_name,
                }
            )
            catalog_kwargs.setdefault(
                "header.X-Iceberg-Access-Delegation", "vended-credentials"
            )
            return ResolvedSource(
                catalog_kwargs=catalog_kwargs,
                table_identifier=f"{schema_name}.{table_name}",
                data_format=ReaderFormat.ICEBERG,
            )

        if reader is ReaderFormat.DELTA:
            if provider != "lakehouse-generic":
                raise ValueError(
                    "Delta access requires a Gravitino 'lakehouse-generic' catalog."
                )
            if (properties.get("format") or "").strip().lower() != "delta":
                raise ValueError(
                    "Gravitino table metadata must declare format='delta'."
                )
            if (properties.get("external") or "").strip().lower() != "true":
                raise ValueError(
                    "Gravitino Delta tables must be registered with external='true'."
                )
        location = properties.get("location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(
                "Gravitino Delta table metadata must include a table-level location."
            )

        return ResolvedSource(path=location, data_format=reader)
