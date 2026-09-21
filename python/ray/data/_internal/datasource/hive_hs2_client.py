"""Bounded-memory HiveServer2 connection adapter built on Impyla internals.

The adapter leaves HS2 request encoding, authentication, sessions, and row
conversion to Impyla. It adds a byte limit at the transport boundary because
Impyla's public fetch size is row-based and therefore cannot bound a variable
width row.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional, Tuple

# A full Thrift response is held by some HiveServer2 transports before row
# decoding begins. Keep this fixed and private: callers should not be able to
# disable the reader's fail-closed memory boundary accidentally.
MAX_HS2_RESPONSE_BYTES = 1024 * 1024
MAX_HS2_SASL_NEGOTIATION_BYTES = 1024 * 1024
MAX_HS2_CONTAINER_ELEMENTS = 65_536
_SUPPORTED_HS2_STACK_VERSIONS = {
    "impyla": "0.24.0",
    "thrift": "0.24.0",
    "thrift-sasl": "0.4.3",
}
_SUPPORTED_HS2_STACK_INSTALL_COMMAND = (
    'pip install "impyla==0.24.0" "thrift==0.24.0" "thrift-sasl==0.4.3"'
)


@dataclass(frozen=True)
class HiveConnectionOptions:
    host: str
    port: int
    auth_mechanism: str
    user: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    kerberos_service_name: str = "hive"
    use_ssl: bool = False
    ca_cert: Optional[str] = None
    timeout: Optional[float] = None


class HiveResponseTooLargeError(RuntimeError):
    """Raised before buffering an HS2 response over a configured protocol limit."""


class HiveClientCompatibilityError(RuntimeError):
    """Raised when the installed Impyla/Thrift stack cannot be bounded safely."""


def _require_supported_hs2_client_stack() -> None:
    for package, supported_version in _SUPPORTED_HS2_STACK_VERSIONS.items():
        try:
            installed_version = version(package)
        except PackageNotFoundError:
            raise ImportError(
                "read_hive requires the validated optional HS2 client stack. "
                f"Install it with `{_SUPPORTED_HS2_STACK_INSTALL_COMMAND}`."
            ) from None
        if installed_version != supported_version:
            raise HiveClientCompatibilityError(
                f"read_hive supports {package}=={supported_version}; "
                f"found {installed_version}. Install the validated stack with "
                f"`{_SUPPORTED_HS2_STACK_INSTALL_COMMAND}`."
            )


class _MessageBudgetTransport:
    """Thrift transport wrapper that bounds one unframed RPC response."""

    def __init__(self, transport: Any, max_bytes: int):
        self._transport = transport
        self._max_bytes = max_bytes
        self._remaining = max_bytes
        self.last_message_bytes = 0

    def begin_message(self) -> None:
        self._remaining = self._max_bytes
        self.last_message_bytes = 0

    def _consume(self, size: int) -> None:
        self._remaining -= size
        self.last_message_bytes += size

    def read(self, size: int) -> bytes:
        if size > self._remaining:
            raise HiveResponseTooLargeError(
                "HiveServer2 response exceeds the configured byte limit"
            )
        data = self._transport.read(size)
        self._consume(len(data))
        return data

    def readAll(self, size: int) -> bytes:
        if size < 0 or size > self._remaining:
            raise HiveResponseTooLargeError(
                "HiveServer2 response exceeds the configured byte limit"
            )
        data = self._transport.readAll(size)
        self._consume(len(data))
        return data

    def write(self, data: bytes) -> None:
        self._transport.write(data)

    def flush(self) -> None:
        self._transport.flush()

    def isOpen(self) -> bool:
        return self._transport.isOpen()

    def open(self) -> None:
        self._transport.open()

    def close(self) -> None:
        self._transport.close()


def _make_bounded_binary_protocol(transport: Any) -> Any:
    """Create a pure-Python Thrift protocol with per-message and field limits.

    The accelerated Impyla protocol bypasses Python's length checks. Using the
    standard protocol ensures declared string/container lengths are rejected
    before Thrift allocates their contents.
    """
    try:
        from thrift.protocol.TBinaryProtocol import TBinaryProtocol
        from thrift.transport.TTransport import TTransportException
    except ImportError as exc:  # pragma: no cover - Impyla requires thrift.
        raise HiveClientCompatibilityError(
            "The installed Impyla package does not provide the expected Thrift protocol"
        ) from exc

    class BoundedBinaryProtocol(TBinaryProtocol):
        def readMessageBegin(self):
            transport.begin_message()
            return super().readMessageBegin()

        def _check_declared_length(self, check_length, length):
            try:
                check_length(length)
            except TTransportException as exc:
                if exc.type == TTransportException.SIZE_LIMIT:
                    raise HiveResponseTooLargeError(
                        "HiveServer2 response exceeds a configured protocol limit"
                    ) from exc
                raise

        def _check_string_length(self, length):
            self._check_declared_length(super()._check_string_length, length)

        def _check_container_length(self, length):
            self._check_declared_length(super()._check_container_length, length)

    return BoundedBinaryProtocol(
        transport,
        string_length_limit=MAX_HS2_RESPONSE_BYTES,
        container_length_limit=MAX_HS2_CONTAINER_ELEMENTS,
    )


def _make_bounded_sasl_transport(
    sasl_client_factory: Any, mechanism: str, socket: Any
) -> Any:
    """Create Impyla's SASL transport with a checked frame length."""
    try:
        from thrift_sasl import TSaslClientTransport
    except ImportError as exc:  # pragma: no cover - Impyla requires thrift-sasl.
        raise HiveClientCompatibilityError(
            "The installed Impyla package does not provide the expected SASL transport"
        ) from exc

    class BoundedSaslClientTransport(TSaslClientTransport):
        def _recv_sasl_message(self) -> Tuple[int, bytes]:
            header = self._trans_read_all(5)
            status, length = struct.unpack(">BI", header)
            if length > MAX_HS2_SASL_NEGOTIATION_BYTES:
                raise HiveResponseTooLargeError(
                    "HiveServer2 SASL negotiation payload exceeds the "
                    "configured byte limit"
                )
            return status, self._trans_read_all(length) if length else b""

        def _read_frame(self) -> None:
            header = self._trans_read_all(4)
            (length,) = struct.unpack(">I", header)
            if length > MAX_HS2_RESPONSE_BYTES:
                raise HiveResponseTooLargeError(
                    "HiveServer2 SASL frame exceeds the configured byte limit"
                )
            payload = self._trans_read_all(length)
            if self.encode:
                success, decoded = self.sasl.decode(header + payload)
                if not success:
                    from thrift.transport.TTransport import TTransportException

                    raise TTransportException(
                        message="HiveServer2 SASL response could not be decoded"
                    )
            else:
                decoded = payload
            # Match thrift-sasl's internal read buffer without importing its
            # implementation-specific BufferIO alias.
            from io import BytesIO

            self._TSaslClientTransport__rbuf = BytesIO(decoded)

    return BoundedSaslClientTransport(sasl_client_factory, mechanism, socket)


def connect_hiveserver2(
    *,
    host: str,
    port: int,
    auth_mechanism: str,
    user: Optional[str],
    password: Optional[str],
    kerberos_service_name: str,
    use_ssl: bool,
    ca_cert: Optional[str],
    timeout: Optional[float],
) -> Tuple[Any, _MessageBudgetTransport]:
    """Open an Impyla HiveServer2 connection with bounded incoming messages."""
    _require_supported_hs2_client_stack()
    try:
        from impala import hiveserver2
        from impala._thrift_api import ThriftClient, get_socket
        from impala.sasl_compat import PureSASLClient
        from thrift.transport.TTransport import TBufferedTransport
    except ImportError as exc:
        raise ImportError(
            "read_hive requires the optional Impyla dependency. Install it in "
            "the runtime environment with "
            f"`{_SUPPORTED_HS2_STACK_INSTALL_COMMAND}`."
        ) from exc

    if not isinstance(auth_mechanism, str):
        raise ValueError("auth_mechanism must be a string")
    mechanism = auth_mechanism.upper()
    if mechanism not in {"NOSASL", "PLAIN", "GSSAPI"}:
        raise ValueError("auth_mechanism must be one of 'NOSASL', 'PLAIN', or 'GSSAPI'")
    if mechanism == "PLAIN" and (not user or not password):
        raise ValueError("PLAIN authentication requires a user and non-empty password")
    if not isinstance(use_ssl, bool):
        raise ValueError("use_ssl must be a bool")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    if ca_cert is not None and not isinstance(ca_cert, str):
        raise ValueError("ca_cert must be a string when provided")
    if ca_cert and not use_ssl:
        raise ValueError("ca_cert requires use_ssl=True")

    # Always verify TLS peer identity. Impyla's public connect() defaults to
    # permissive verification, which is not suitable for this datasource.
    socket = get_socket(
        host,
        port,
        use_ssl=use_ssl,
        ca_cert=ca_cert,
        verify_cert=True,
    )
    if timeout is not None:
        socket.setTimeout(timeout * 1000)

    if mechanism == "NOSASL":
        transport = TBufferedTransport(socket)
    else:

        def sasl_client_factory():
            return PureSASLClient(
                host,
                username=user,
                password=password,
                service=kerberos_service_name,
            )

        transport = _make_bounded_sasl_transport(sasl_client_factory, mechanism, socket)

    try:
        transport.open()
        budget_transport = _MessageBudgetTransport(transport, MAX_HS2_RESPONSE_BYTES)
        protocol = _make_bounded_binary_protocol(budget_transport)
        service = hiveserver2.HS2Service(ThriftClient(protocol), retries=3)
        connection = hiveserver2.HiveServer2Connection(service, default_db=None)
        return connection, budget_transport
    except Exception:
        try:
            transport.close()
        except Exception:
            pass
        raise
