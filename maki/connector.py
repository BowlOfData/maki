"""
The hardened HTTP layer for the Maki framework.

Every outbound HTTP request in maki — backends, the distributed proxy, and
plugins — goes through this module so that URL validation, SSRF protection,
timeouts, and error classification live in exactly one place.

Two clients are provided:

``Connector``
    Synchronous client built on ``requests.Session``. Supports streaming,
    custom TLS (verify/cert), default headers, and per-request overrides.

``AsyncConnector``
    Asynchronous client built on ``httpx.AsyncClient`` for the async
    backend paths.

SSRF protection
---------------
When ``ssrf_protect=True`` (the default) and ``allow_private=False``:

* the URL scheme must be http/https and the hostname passes the static
  checks in :meth:`Utils._validate_domain` (blacklist, private IP literals);
* hostnames are resolved at **connection time** and every resolved address
  is validated against the private/reserved ranges; the connection is then
  pinned to the validated IP, so a DNS rebind between validation and
  connect cannot redirect the request (TLS still verifies against the
  original hostname via SNI).

``allow_private=True`` is for operator-configured endpoints (a local or
LAN Ollama daemon, a registered remote agent): the scheme and hostname
format are still validated, but private/reserved addresses are permitted
and no DNS pinning is applied. Loopback is always allowed.

Error classification
--------------------
Transport failures and (optionally) HTTP error statuses are mapped onto
the Maki exception tree:

    timeout (transport or HTTP 408/504)  -> MakiTimeoutError
    connection / transport failure, 5xx  -> MakiNetworkError
    other 4xx                            -> MakiAPIError
    invalid URL / blocked address        -> MakiValidationError
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse
from typing import Any, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from .config import DEFAULT_REQUEST_TIMEOUT
from .exceptions import (
    MakiAPIError,
    MakiError,
    MakiNetworkError,
    MakiTimeoutError,
    MakiValidationError,
)
from .utils import Utils

logger = logging.getLogger(__name__)

# How much of an error-response body to include in raised exceptions.
_DETAIL_LIMIT = 500


def validate_url(url: str, allow_private: bool = False) -> None:
    """Static URL checks (scheme, hostname format, blocked literals).

    Raises:
        MakiValidationError: if the URL is malformed or blocked.
    """
    if not isinstance(url, str) or not url.strip():
        raise MakiValidationError("URL must be a non-empty string")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise MakiValidationError("URL must use http or https protocol")
    if not parsed.hostname:
        raise MakiValidationError(f"URL has no hostname: {url!r}")
    try:
        Utils._validate_domain(parsed.hostname, allow_private=allow_private)
    except ValueError as e:
        raise MakiValidationError(str(e)) from e


# ---------------------------------------------------------------------------
# DNS resolution + validation (shared by the pinned adapter and AsyncConnector)
# ---------------------------------------------------------------------------

def _resolve_and_validate(host: str, port: Optional[int]) -> list:
    """Resolve *host* and validate every resolved address.

    Returns the list of validated addresses in resolution order; the
    caller pins the connection to one of them. IP literals are validated
    directly and returned as a single-element list.

    Raises:
        MakiValidationError: if any resolved address is private/reserved.
        MakiNetworkError: if the hostname does not resolve.
    """
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        ip = None
    if ip is not None:
        try:
            Utils._validate_ip(ip, host)
        except ValueError as e:
            raise MakiValidationError(str(e)) from e
        return [host]

    try:
        infos = socket.getaddrinfo(bare, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise MakiNetworkError(f"DNS resolution failed for '{host}': {e}") from e
    if not infos:
        raise MakiNetworkError(f"DNS resolution returned no addresses for '{host}'")

    # Reject the whole host if *any* resolved address is private: a mix of
    # public and private records is the classic rebinding/round-robin trick.
    addresses = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            Utils._validate_ip(ipaddress.ip_address(addr), f"{host} -> {addr}")
        except ValueError as e:
            raise MakiValidationError(str(e)) from e
        if addr not in addresses:
            addresses.append(addr)
    return addresses


# ---------------------------------------------------------------------------
# Connect-time pinning for the sync (requests/urllib3) engine
# ---------------------------------------------------------------------------

def _connect_pinned(conn, super_new_conn) -> socket.socket:
    """Resolve+validate, then connect with per-address fallback.

    Every attempted address has passed validation, so falling back through
    the resolution list (e.g. an IPv6 record on a dual-stack host whose
    service listens on IPv4 only) keeps the security property intact.
    """
    addresses = _resolve_and_validate(conn._dns_host, conn.port)
    last_exc: Exception = OSError("no addresses to connect to")
    for addr in addresses:
        conn._dns_host = addr
        try:
            return super_new_conn()
        except Exception as e:
            last_exc = e
    raise last_exc


class _PinnedHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        return _connect_pinned(self, lambda: HTTPConnection._new_conn(self))


class _PinnedHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        # Preserve the original hostname for SNI / certificate matching
        # before the socket target is swapped to the validated IP.
        if self.server_hostname is None:
            self.server_hostname = self.host
        return _connect_pinned(self, lambda: HTTPSConnection._new_conn(self))


class _PinnedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PinnedHTTPConnection


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection


class _SSRFAdapter(HTTPAdapter):
    """requests adapter whose connections resolve+validate+pin at connect time."""

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PinnedHTTPConnectionPool,
            "https": _PinnedHTTPSConnectionPool,
        }


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class Connector:
    """Hardened synchronous HTTP client (see module docstring).

    Args:
        timeout:       Default timeout in seconds (or a (connect, read)
                       tuple). Falls back to ``DEFAULT_REQUEST_TIMEOUT``.
        ssrf_protect:  Validate URLs before every request.
        allow_private: Permit private/reserved target addresses and skip
                       DNS pinning (for operator-configured endpoints).
        headers:       Default headers applied to every request.
        verify:        TLS verification: True, False, or a CA-bundle path.
        cert:          Client certificate (path or (cert, key) tuple).
    """

    def __init__(
        self,
        timeout: Union[int, float, tuple, None] = None,
        ssrf_protect: bool = True,
        allow_private: bool = False,
        headers: Optional[dict] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Any] = None,
    ) -> None:
        self.timeout = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
        self._ssrf_protect = ssrf_protect
        self._allow_private = allow_private
        self._session = requests.Session()
        if headers:
            self._session.headers.update(headers)
        if verify is not True:
            self._session.verify = verify
        if cert is not None:
            self._session.cert = cert
        if ssrf_protect and not allow_private:
            adapter = _SSRFAdapter()
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

    # -- validation -----------------------------------------------------

    def validate_url(self, url: str) -> None:
        """Static URL checks; see the module-level :func:`validate_url`."""
        validate_url(url, allow_private=self._allow_private)

    # -- request core ---------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        raise_on_status: bool = True,
        timeout: Union[int, float, tuple, None] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a request with validation and error classification.

        Extra kwargs (``json``, ``params``, ``headers``, ``stream``,
        ``allow_redirects``, …) are passed to ``requests``.

        Args:
            raise_on_status: When True (default), non-2xx responses raise
                the mapped Maki exception. When False the response is
                returned for the caller to inspect.

        Raises:
            MakiValidationError, MakiTimeoutError, MakiNetworkError,
            MakiAPIError.
        """
        if self._ssrf_protect:
            self.validate_url(url)
        kwargs.setdefault("timeout", timeout if timeout is not None else self.timeout)
        # Dispatch through the named session method (session.get/post/...)
        # rather than session.request so that test suites patching
        # requests.Session.post/get keep intercepting these calls.
        sender = getattr(self._session, method.lower(), None)
        try:
            if sender is not None:
                response = sender(url, **kwargs)
            else:
                response = self._session.request(method, url, **kwargs)
        except MakiError:
            raise  # raised by the pinned adapter at connect time
        except requests.exceptions.Timeout as e:
            raise MakiTimeoutError(f"{method} {url} timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise MakiNetworkError(f"{method} {url} connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            raise MakiNetworkError(f"{method} {url} request failed: {e}") from e
        if raise_on_status:
            self.raise_for_response(response)
        return response

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> requests.Response:
        return self.request("DELETE", url, **kwargs)

    # -- response helpers -------------------------------------------------

    @staticmethod
    def raise_for_response(response) -> None:
        """Map a non-2xx response onto the Maki exception tree.

        Works with any response object exposing ``status_code`` and
        ``text`` (both ``requests`` and ``httpx``).
        """
        status = getattr(response, "status_code", None)
        if not isinstance(status, int) or status < 400:
            return
        try:
            detail = response.text[:_DETAIL_LIMIT]
        except Exception:
            detail = "<unreadable body>"
        if status in (408, 504):
            raise MakiTimeoutError(f"HTTP {status} (timeout): {detail}")
        if status >= 500:
            raise MakiNetworkError(f"HTTP server error {status}: {detail}")
        raise MakiAPIError(f"HTTP client error {status}: {detail}")

    @staticmethod
    def iter_lines(response, **kwargs):
        """Iterate a streaming response's lines with error classification.

        Mid-stream transport failures (the request itself succeeded but the
        body was cut off or timed out) surface as raw ``requests``
        exceptions from ``Response.iter_lines``; this wrapper maps them
        onto the Maki exception tree like the request path does.
        """
        try:
            yield from response.iter_lines(**kwargs)
        except requests.exceptions.Timeout as e:
            raise MakiTimeoutError(f"Stream timed out mid-body: {e}") from e
        except requests.exceptions.RequestException as e:
            raise MakiNetworkError(f"Stream failed mid-body: {e}") from e

    @staticmethod
    def json_or_raise(response) -> Any:
        """Parse a response body as JSON.

        Raises:
            MakiAPIError: if the body is not valid JSON.
        """
        try:
            return response.json()
        except ValueError as e:
            raise MakiAPIError(f"Invalid JSON in API response: {e}") from e

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Connector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            session = getattr(self, "_session", None)
            if session is not None:
                session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class AsyncConnector:
    """Hardened asynchronous HTTP client built on httpx.

    A fresh ``httpx.AsyncClient`` is created per request, so instances are
    safe to share across event loops; the lost connection pooling is
    acceptable for the current async surface (single-shot backend calls).

    Note: the async engine validates resolved addresses *before* the
    request (httpx exposes no connect-time DNS hook), so a rebind in the
    window between validation and connect is not pinned away as it is on
    the sync path. Use the sync ``Connector`` where that matters.
    """

    def __init__(
        self,
        timeout: Union[int, float, None] = None,
        ssrf_protect: bool = True,
        allow_private: bool = False,
        headers: Optional[dict] = None,
    ) -> None:
        self.timeout = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
        self._ssrf_protect = ssrf_protect
        self._allow_private = allow_private
        self._headers = dict(headers) if headers else None

    async def request(
        self,
        method: str,
        url: str,
        *,
        raise_on_status: bool = True,
        timeout: Union[int, float, None] = None,
        **kwargs,
    ):
        import httpx

        if self._ssrf_protect:
            validate_url(url, allow_private=self._allow_private)
            if not self._allow_private:
                parsed = urllib.parse.urlparse(url)
                _resolve_and_validate(parsed.hostname or "", parsed.port)
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            async with httpx.AsyncClient(
                timeout=effective_timeout, headers=self._headers
            ) as client:
                response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"{method} {url} timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"{method} {url} connection failed: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"{method} {url} request failed: {e}") from e
        if raise_on_status:
            Connector.raise_for_response(response)
        return response

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs):
        return await self.request("POST", url, **kwargs)
