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
        "198.51.100.0/24", # TEST-NET-2
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

    @staticmethod
    def _validate_ip(
        ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
        label: str,
    ) -> None:
        """Reject private/reserved IP addresses (loopback is allowed).

        Args:
            ip:    The parsed IP address to check.
            label: Human-readable name for error messages (the original
                   host string, or "host -> addr" for resolved addresses).

        Raises:
            ValueError: If the address is private, link-local, or reserved.
        """
        if ip.is_loopback:
            return
        for ip_range in Utils.PRIVATE_IP_RANGES:
            if ip in ipaddress.ip_network(ip_range):
                raise ValueError(f"Access to private IP address '{label}' is not allowed")
        if ip.is_link_local or ip.is_reserved:
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
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', domain):
            raise ValueError(f"Invalid domain format: {domain}")

        if '..' in domain:
            raise ValueError("Domain contains invalid pattern")

        if re.search(r'[^\w\.\-\:]', domain):
            raise ValueError("Domain contains invalid characters")

        if len(domain) > 253:
            raise ValueError("Domain name too long")

        labels = domain.split('.')
        for label in labels:
            if len(label) > 63:
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
                # For sync methods, we can close directly
                if hasattr(client, 'close'):
                    client.close()
                # For async clients, run aclose() via the event loop
                elif hasattr(client, 'aclose'):
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        # A loop is already running; schedule cleanup as a
                        # fire-and-forget task, holding a strong reference
                        # until it completes.
                        task = loop.create_task(client.aclose())
                        _CLEANUP_TASKS.add(task)
                        task.add_done_callback(_CLEANUP_TASKS.discard)
                    except RuntimeError:
                        # No running event loop — safe to block
                        asyncio.run(client.aclose())
            except Exception as e:
                logger.debug(f"Failed to close client: {e}")
