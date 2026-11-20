"""Shared TLS and certificate-bound identity helpers for internal gRPC hops.

The controller, workers, and aggregation server use this module for two things:

* converting component config into gRPC channel/server credentials, and
* deriving a caller's role and identity from the URI SANs of its client
  certificate.

Only the standard library and ``grpc`` are used. The URI SAN schema is::

    exaflow://<federation>/controller
    exaflow://<federation>/worker/<worker_id>

A controller certificate carries exactly one ``exaflow://`` URI SAN; a worker
certificate carries one with its worker id. ``worker_id`` is what the
aggregation server matches against the ``ConfigureRequest.worker_ids`` allow
list, so a stolen or mis-issued certificate cannot claim another worker's
submissions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Mapping
from typing import Optional

import grpc

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"true", "1", "yes"}
_FALSE_VALUES = {"false", "0", "no"}
_URI_PREFIX = "exaflow://"


class UnauthenticatedError(Exception):
    """The peer certificate does not establish an Exaflow identity."""


class UnauthorizedError(Exception):
    """The peer identity is valid but not permitted for this call."""


class DuplicateWorkerSubmissionError(Exception):
    """The same worker submitted twice for one aggregation step."""


@dataclass(frozen=True)
class GrpcTlsConfig:
    """TLS settings shared by the aggregation server and its clients."""

    enabled: bool = False
    ca_cert_path: str | None = None
    cert_path: str | None = None
    key_path: str | None = None
    require_client_auth: bool = True
    target_name_override: str | None = None


@dataclass(frozen=True)
class PeerIdentity:
    """Identity extracted from a client certificate's URI SAN."""

    role: Literal["controller", "worker"]
    federation: str
    worker_id: Optional[str]
    uri: str


def parse_tls_bool(value: object, default: bool = False) -> bool:
    """Parse a config/config-file boolean without accepting silent typos.

    ``None`` and a blank string fall back to ``default``. Bare booleans are
    returned as-is; the usual ``true``/``false`` string spellings (and ``1``
    and ``0``) are accepted case-insensitively. Anything else is a config
    error rather than a surprising ``False``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return default
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
    raise ValueError(f"Unrecognized TLS boolean value: {value!r}")


def _optional_path(raw: Mapping[str, object], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_tls_config(
    raw: Mapping[str, object] | None,
    *,
    require_client_auth_default: bool = True,
) -> GrpcTlsConfig:
    """Build a :class:`GrpcTlsConfig` from a config mapping.

    Empty and whitespace-only paths become ``None`` so that an unset
    environment variable (which config files interpolate to ``""``) does not
    turn into a path to the current directory.
    """
    values: Mapping[str, object] = raw or {}
    return GrpcTlsConfig(
        enabled=parse_tls_bool(values.get("enabled"), False),
        ca_cert_path=_optional_path(values, "ca_cert_path"),
        cert_path=_optional_path(values, "cert_path"),
        key_path=_optional_path(values, "key_path"),
        require_client_auth=parse_tls_bool(
            values.get("require_client_auth"), require_client_auth_default
        ),
        target_name_override=_optional_path(values, "target_name_override"),
    )


def build_server_credentials(tls: GrpcTlsConfig) -> grpc.ServerCredentials:
    """Read PEM files and build credentials for the aggregation server."""
    if not tls.cert_path or not tls.key_path:
        raise ValueError("TLS server credentials require cert_path and key_path")
    if tls.require_client_auth and not tls.ca_cert_path:
        raise ValueError("TLS server credentials require ca_cert_path for client auth")
    key_bytes = Path(tls.key_path).read_bytes()
    cert_bytes = Path(tls.cert_path).read_bytes()
    ca_bytes = Path(tls.ca_cert_path).read_bytes() if tls.ca_cert_path else None
    return grpc.ssl_server_credentials(
        [(key_bytes, cert_bytes)],
        root_certificates=ca_bytes,
        require_client_auth=tls.require_client_auth,
    )


def build_channel_credentials(tls: GrpcTlsConfig) -> grpc.ChannelCredentials:
    """Read PEM files and build credentials for a controller/worker channel."""
    if not tls.ca_cert_path:
        raise ValueError("TLS channel credentials require ca_cert_path")
    if bool(tls.cert_path) != bool(tls.key_path):
        raise ValueError(
            "TLS channel credentials require both cert_path and key_path, or neither"
        )
    ca_bytes = Path(tls.ca_cert_path).read_bytes()
    cert_bytes = Path(tls.cert_path).read_bytes() if tls.cert_path else None
    key_bytes = Path(tls.key_path).read_bytes() if tls.key_path else None
    return grpc.ssl_channel_credentials(
        root_certificates=ca_bytes,
        private_key=key_bytes,
        certificate_chain=cert_bytes,
    )


def channel_options_with_tls(
    tls: GrpcTlsConfig,
    base_options: list[tuple[str, object]] | None = None,
) -> list[tuple[str, object]]:
    """Return channel options with the optional SAN name override appended."""
    options = list(base_options or [])
    if tls.enabled and tls.target_name_override:
        options.append(("grpc.ssl_target_name_override", tls.target_name_override))
    return options


def _as_text(value: object) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return None


def _peer_uris(auth_context: Mapping[str, object]) -> list[str]:
    """Collect ``exaflow://`` URI SANs advertised by the peer certificate.

    Only reads ``peer_uri`` (strictly URI SAN entries populated by grpcio),
    never the mixed ``x509_subject_alternative_name`` list where DNS/email SANs
    could otherwise spoof an identity URI.
    """
    return [
        text
        for value in auth_context.get("peer_uri", ()) or ()
        if (text := _as_text(value)) is not None and text.startswith(_URI_PREFIX)
    ]


def _parse_identity_uri(uri: str) -> PeerIdentity:
    if not uri.startswith(_URI_PREFIX):
        raise UnauthenticatedError(f"Unexpected identity URI: {uri!r}")
    remainder = uri[len(_URI_PREFIX) :]
    if "?" in remainder or "#" in remainder:
        raise UnauthenticatedError(
            f"Identity URI must not have a query or fragment: {uri!r}"
        )
    federation, separator, path = remainder.partition("/")
    if not separator or not federation:
        raise UnauthenticatedError(f"Malformed identity URI: {uri!r}")
    # The federation occupies the authority component. Reject userinfo, a port,
    # and whitespace rather than letting them become part of the identity.
    if any(char in federation for char in ":@ \t\r\n/\\"):
        raise UnauthenticatedError(f"Malformed federation in identity URI: {uri!r}")
    segments = path.split("/")
    if not segments or any(segment == "" for segment in segments):
        raise UnauthenticatedError(f"Identity URI has empty path segments: {uri!r}")
    if segments == ["controller"]:
        return PeerIdentity("controller", federation, None, uri)
    if len(segments) == 2 and segments[0] == "worker":
        worker_id = segments[1]
        if not worker_id.strip() or any(char in worker_id for char in ":@ \t\r\n/\\"):
            raise UnauthenticatedError(f"Malformed worker_id in identity URI: {uri!r}")
        return PeerIdentity("worker", federation, worker_id, uri)
    raise UnauthenticatedError(f"Unrecognized identity URI shape: {uri!r}")


def extract_peer_identity(
    context: grpc.ServicerContext,
    *,
    expected_federation: str | None = None,
) -> PeerIdentity:
    """Derive the caller's role/federation/worker id from its mTLS certificate.

    Raises :class:`UnauthenticatedError` when there is no TLS peer identity or
    the URI SAN is ambiguous/malformed, and :class:`UnauthorizedError` when the
    identity belongs to another federation.
    """
    auth_context = context.auth_context() or {}
    security_types = [
        _as_text(value) or ""
        for value in auth_context.get("transport_security_type", ()) or ()
    ]
    if not any("ssl" in security_type.lower() for security_type in security_types):
        raise UnauthenticatedError("Mutual TLS is required for this RPC")

    uris = _peer_uris(auth_context)
    if len(uris) != 1:
        raise UnauthenticatedError(
            "Peer certificate must contain exactly one exaflow:// identity URI"
        )
    identity = _parse_identity_uri(uris[0])
    if expected_federation and identity.federation != expected_federation:
        raise UnauthorizedError(
            f"Certificate federation '{identity.federation}' does not match "
            f"expected federation '{expected_federation}'"
        )
    return identity
