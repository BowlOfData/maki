import json
import base64
import os
import logging
import re
import ipaddress
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Strong references to in-flight client-cleanup tasks. asyncio keeps only
# weak references to tasks, so a fire-and-forget create_task() result can be
# garbage-collected before it ever runs.
_CLEANUP_TASKS: set = set()


class Utils:

    # List of private IP ranges that should be blocked to prevent SSRF
    PRIVATE_IP_RANGES = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fe80::/10",
        "169.254.0.0/16",  # Link-local addresses
        "100.64.0.0/10",   # Shared address space (CGNAT)
        "192.0.0.0/24",    # IETF Protocol Assignments
        "192.0.2.0/24",    # TEST-NET-1
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "198.18.0.0/15",   # Benchmarking (RFC 2544)
        "240.0.0.0/4",     # Reserved for future use
        "0.0.0.0/8",       # This network
        "::/128",          # Unspecified address
        "fc00::/7",        # Unique local addresses
    ]

    # Blacklisted domains that should be blocked
    BLACKLISTED_DOMAINS = [
        "0.0.0.0",
        "255.255.255.255"
    ]

    # IPv6 prefix that embeds an IPv4 address in its last 32 bits (RFC 6052 NAT64).
    _NAT64 = ipaddress.ip_network("64:ff9b::/96")

    @staticmethod
    def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list:
        """IPv4 addresses an IPv6 address stands for: ``::ffff:a.b.c.d``, 6to4 (``2002:aabb:ccdd::``), NAT64
        and the deprecated IPv4-compatible form. A guard that only looked at the IPv6 shell would let
        ``2002:7f00:1::`` (127.0.0.1) or ``64:ff9b::a9fe:a9fe`` (the cloud metadata address) through."""
        found = []
        if ip.ipv4_mapped is not None:
            found.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            found.append(ip.sixtofour)
        if ip in Utils._NAT64:
            found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if 1 < int(ip) < 2 ** 32:  # ::a.b.c.d (IPv4-compatible); :: and ::1 are judged as themselves
            found.append(ipaddress.IPv4Address(int(ip)))
        return found

    @staticmethod
    def _validate_ip(
        ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
        label: str,
    ) -> None:
        """Accept only globally routable addresses.

        Refused: loopback, private, link-local, shared (CGNAT), multicast, unspecified, documentation,
        benchmarking, reserved and other special-purpose ranges (everything ``ipaddress`` does not call
        global), IPv6 forms that embed an IPv4 address (mapped, 6to4, NAT64, IPv4-compatible) when that
        address is refused (mapped and NAT64 forms of a public address stay allowed: that address is what is
        reached), and Teredo tunnels. Loopback is refused too: a URL a model or a web page can
        influence must not reach services on this machine (Ollama, a database console, an unauthenticated
        ``maki serve``). Operators who need it set ``MAKI_ALLOW_LOOPBACK_URLS=1``; endpoints they configure
        themselves (Ollama, llama.cpp, vLLM, remote agents) are not checked here at all (``allow_private=True``).

        Args:
            ip:    The parsed IP address to check.
            label: Human-readable name for error messages (the original
                   host string, or "host -> addr" for resolved addresses).

        Raises:
            ValueError: If the address, or an IPv4 address it embeds, is not publicly routable.
        """
        if ip.version == 6 and ip.teredo is not None:
            raise ValueError(f"Access to Teredo tunnel address '{label}' is not allowed")
        if ip.version == 6 and (ip.ipv4_mapped is not None or ip in Utils._NAT64):
            # translation forms reach the IPv4 address they carry, so that address is what is judged
            candidates = Utils._embedded_ipv4(ip)[:1] if ip.ipv4_mapped is not None else Utils._embedded_ipv4(ip)
        else:  # 6to4 and the IPv4-compatible form: the IPv6 range itself is refused as well as what it embeds
            candidates = [ip] + (Utils._embedded_ipv4(ip) if ip.version == 6 else [])
        for candidate in candidates:
            Utils._validate_single_ip(candidate, label)

    @staticmethod
    def _validate_single_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address], label: str) -> None:
        if ip.is_loopback:
            if os.environ.get("MAKI_ALLOW_LOOPBACK_URLS", "").strip().lower() in ("1", "true", "yes", "on"):
                return
            raise ValueError(f"Access to loopback address '{label}' is not allowed "
                             "(set MAKI_ALLOW_LOOPBACK_URLS=1 to permit it)")
        for ip_range in Utils.PRIVATE_IP_RANGES:
            if ip in ipaddress.ip_network(ip_range):
                raise ValueError(f"Access to private IP address '{label}' is not allowed")
        if ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified or not ip.is_global:
            raise ValueError(f"Access to special IP address '{label}' is not allowed")

    @staticmethod
    def _validate_domain(domain: str, allow_private: bool = False) -> None:
        """Validate domain name to prevent SSRF attacks

        Args:
            domain: The domain name to validate
            allow_private: Permit private/reserved IP literals (for
                operator-configured endpoints such as a LAN Ollama host).
                Format and blacklist checks still apply.

        Raises:
            ValueError: If domain is invalid or potentially malicious
        """
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("Domain must be a non-empty string")

        domain = domain.strip()

        # Check for blacklisted domains
        if domain.lower() in Utils.BLACKLISTED_DOMAINS:
            raise ValueError(f"Access to domain '{domain}' is not allowed")

        # Try to parse as an IP address first. Separate the parse attempt from
        # the security checks so that security-raised ValueErrors are never
        # swallowed by the except clause (the original bug: inner
        # `raise ValueError(...)` was caught by the outer `except ValueError`,
        # bypassing the intended block and falling through to domain validation).
        clean_domain = domain
        if domain.startswith('[') and domain.endswith(']'):
            clean_domain = domain[1:-1]

        try:
            ip = ipaddress.ip_address(clean_domain)
            is_ip = True
        except ValueError:
            is_ip = False

        if is_ip:
            if not allow_private:
                Utils._validate_ip(ip, domain)
            return  # valid (or explicitly permitted) IP

        # Not an IP address — validate as a domain name
        Utils._validate_domain_name(domain)

    @staticmethod
    def _validate_domain_name(domain: str) -> None:
        """Format checks for a hostname that is not an IP literal."""
        label = r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
        if not re.match(rf'^{label}(\.{label})*$', domain):
            raise ValueError(f"Invalid domain format: {domain}")

        if '..' in domain:
            raise ValueError("Domain contains invalid pattern")

        if re.search(r'[^\w\.\-\:]', domain):
            raise ValueError("Domain contains invalid characters")

        if len(domain) > 253:
            raise ValueError("Domain name too long")

        if any(len(part) > 63 for part in domain.split('.')):
            raise ValueError("Domain label too long")

    @staticmethod
    def jsonify(data) -> Any:
        """Parse JSON data

        Args:
            data: JSON string to parse

        Returns:
            Parsed JSON object

        Raises:
            ValueError: If data is not a valid JSON string
        """
        logger = logging.getLogger(__name__)

        if not isinstance(data, str):
            raise ValueError("Data must be a string")

        if not data.strip():
            raise ValueError("Data cannot be empty")

        try:
            result = json.loads(data)
            logger.debug("JSON parsing completed successfully")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise ValueError(f"Invalid JSON data: {str(e)}") from e

    @staticmethod
    def convert64(img: str, allowed_dirs: Optional[list[str]] = None) -> str:
        """Convert an image file to a base64-encoded string

        Args:
            img: path to image file
            allowed_dirs: optional list of base directories the image must
                live in after full symlink resolution. When omitted, any
                readable file is accepted.

        Returns:
            Base64-encoded string of the image contents

        Raises:
            ValueError: If img is not a valid string, points to a directory,
                or resolves outside allowed_dirs
            FileNotFoundError: If the file doesn't exist
            OSError: For other file reading errors
        """
        logger = logging.getLogger(__name__)

        if not isinstance(img, str) or not img.strip():
            raise ValueError("Image path must be a non-empty string")

        img = img.strip()

        # Resolve symlinks and '..' fully, then verify containment in an
        # allowed base directory (same pattern as FileWriter._safe_path).
        # A bare symlink anywhere in the ancestry is not an attack — on macOS
        # /tmp and /var are symlinks — escaping the allowed base is.
        real_img = os.path.realpath(img)
        if allowed_dirs is not None:
            bases = [os.path.realpath(d) for d in allowed_dirs]
            if not any(real_img == b or real_img.startswith(b + os.sep) for b in bases):
                raise ValueError(
                    f"Image path '{img}' resolves outside the allowed directories"
                )

        if not os.path.exists(real_img):
            raise FileNotFoundError(f"Image file not found: {img}")

        if not os.path.isfile(real_img):
            raise ValueError("Image path must point to a file, not a directory")

        try:
            logger.debug(f"Converting image to base64: {img}")
            with open(real_img, "rb") as image_file:
                result = base64.b64encode(image_file.read()).decode("ascii")
            logger.debug("Image conversion completed successfully")
            return result
        except OSError as e:
            logger.error(f"Image conversion failed: {str(e)}", exc_info=True)
            raise OSError(f"Error reading image file {img}: {str(e)}") from e

    @staticmethod
    def cleanup_response(response, client=None):
        """
        Common utility function to clean up HTTP responses and clients.

        This function handles proper cleanup of both requests.Response objects
        and httpx.AsyncClient instances to prevent resource leaks.

        Args:
            response: requests.Response object or None
            client: httpx.AsyncClient instance or None
        """
        # Clean up requests.Response object if provided
        if response is not None and hasattr(response, 'close'):
            try:
                response.close()
            except Exception as e:
                logger.debug(f"Failed to close response: {e}")

        # Clean up httpx.AsyncClient if provided
        if client is not None:
            try:
                Utils._close_client(client)
            except Exception as e:
                logger.debug(f"Failed to close client: {e}")

    @staticmethod
    def _close_client(client) -> None:
        # For sync methods, we can close directly
        if hasattr(client, 'close'):
            client.close()
        # For async clients, run aclose() via the event loop
        elif hasattr(client, 'aclose'):
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop — safe to block
                asyncio.run(client.aclose())
                return
            # A loop is already running; schedule cleanup as a fire-and-forget
            # task, holding a strong reference until it completes.
            task = loop.create_task(client.aclose())
            _CLEANUP_TASKS.add(task)
            task.add_done_callback(_CLEANUP_TASKS.discard)
