import json
import base64
import os
import logging
import re
import ipaddress
import socket
from .urls import GENERIC_LLAMA_URL

class Utils:

    # List of private IP ranges that should be blocked to prevent SSRF
    PRIVATE_IP_RANGES = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fe80::/10",
        "169.254.0.0/16",  # Link-local addresses
        "100.64.0.0/10",   # Shared address space
        "192.0.0.0/24",    # IETF Protocol Assignments
        "192.0.2.0/24",    # TEST-NET-1
        "198.18.0.0/15",   # TEST-NET-2
        "198.51.100.0/24", # TEST-NET-3
        "203.0.113.0/24",  # TEST-NET-4
        "240.0.0.0/4",     # Reserved for future use
        "0.0.0.0/8",       # This network
        "128.0.0.0/16",    # Reserved for future use
        "::/128",          # Unspecified address
        "fc00::/7",        # Unique local addresses
    ]

    # Blacklisted domains that should be blocked
    BLACKLISTED_DOMAINS = [
        "0.0.0.0",
        "255.255.255.255"
    ]

    @staticmethod
    def _validate_domain(domain: str) -> None:
        """Validate domain name to prevent SSRF attacks

        Args:
            domain: The domain name to validate

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
            # Loopback addresses are allowed for local development
            if ip.is_loopback:
                return
            for ip_range in Utils.PRIVATE_IP_RANGES:
                if ip in ipaddress.ip_network(ip_range):
                    raise ValueError(f"Access to private IP address '{domain}' is not allowed")
            if ip.is_link_local or ip.is_reserved:
                raise ValueError(f"Access to special IP address '{domain}' is not allowed")
            return  # valid public IP

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
    def _validate_port(port: str) -> None:
        """Validate port number to prevent invalid values

        Args:
            port: The port number to validate

        Raises:
            ValueError: If port is invalid
        """
        if not isinstance(port, str) or not port.strip():
            raise ValueError("Port must be a non-empty string")

        port = port.strip()

        # Check if it's a valid numeric port
        if not re.match(r'^[0-9]+$', port):
            raise ValueError("Port must be a valid numeric string")

        port_num = int(port)
        if port_num < 1 or port_num > 65535:
            raise ValueError("Port must be between 1 and 65535")

        # Additional check to prevent certain problematic ports
        if port_num in [0, 80, 443, 22, 21]:  # Common ports that might be problematic
            # These are allowed but with warning
            logger = logging.getLogger(__name__)
            logger.warning(f"Using potentially problematic port: {port_num}")

    @staticmethod
    def compose_url(url: str, port: str, action: str) -> str:
        """Compose the full URL for the Ollama API endpoint

        Args:
            url: the domain or IP address
            port: the port number
            action: the API action (e.g., 'generate', 'version')

        Returns:
            The complete URL string

        Raises:
            ValueError: If any parameter is invalid
        """
        logger = logging.getLogger(__name__)

        # Validate inputs
        # For URLs with protocols, extract just the domain part for validation
        original_url = url.strip()
        if original_url.startswith(('http://', 'https://')):
            # Extract domain part for validation
            import urllib.parse
            parsed = urllib.parse.urlparse(original_url)
            domain_part = parsed.hostname or parsed.netloc
            if domain_part:
                # Handle IPv6 addresses with brackets properly
                if domain_part.startswith('[') and domain_part.endswith(']'):
                    # Remove brackets for validation but keep them for later use
                    domain_part = domain_part[1:-1]
                Utils._validate_domain(domain_part)
        else:
            # If no protocol is provided, default to http:// for better compatibility
            # with the examples in the README
            Utils._validate_domain(original_url)

        Utils._validate_port(port)

        if not isinstance(action, str) or not action.strip():
            raise ValueError("Action must be a non-empty string")

        # Sanitize inputs
        url = url.strip()
        port = port.strip()
        action = action.strip()

        # Extract and preserve any explicit protocol BEFORE sanitization.
        # The sanitization regex removes '/' characters, which would destroy
        # 'http://' or 'https://' if left in the url string.
        import urllib.parse
        protocol = "http"  # default protocol
        domain_part = url
        if url.lower().startswith(('http://', 'https://')):
            parsed = urllib.parse.urlparse(url)
            protocol = parsed.scheme  # 'http' or 'https'
            domain_part = parsed.hostname or parsed.netloc
            # Strip surrounding brackets from IPv6 addresses (e.g. '[::1]' -> '::1')
            if domain_part.startswith('[') and domain_part.endswith(']'):
                domain_part = domain_part[1:-1]
        else:
            # If no protocol is provided in the input, we still need to ensure
            # a proper protocol is used for the final URL
            # This addresses the issue where README examples use "localhost"
            # without explicit protocol
            pass

        # Sanitize only the domain portion — keep alphanumerics, dots, hyphens, colons.
        # Forward slashes are intentionally excluded here; the protocol is handled separately.
        domain_part = re.sub(r'[^a-zA-Z0-9.\-:]', '', domain_part)
        action = re.sub(r'[^a-zA-Z0-9\-_/.]', '', action)

        # Guard: path traversal in action
        if '/../' in action or '..\\' in action or '..' in action:
            raise ValueError("Action contains invalid path traversal characters")

        # Guard: URL-encoding abuse (% not expected in a bare domain)
        if '%' in domain_part:
            raise ValueError("Invalid characters in URL")

        # Guard: port must still be purely numeric after earlier validation
        if not re.match(r'^[0-9]+$', port):
            raise ValueError("Port must be a valid numeric string")

        # Guard: ensure the sanitized domain contains only safe characters
        if not re.match(r'^[a-zA-Z0-9.\-:]+$', domain_part):
            raise ValueError("Invalid domain format after sanitization")

        composed = GENERIC_LLAMA_URL.format(domain=domain_part, port=port, action=action)
        # Always enforce the correct protocol by replacing whatever the template
        # may already contain (e.g. a hardcoded 'http://') with the protocol that
        # was explicitly supplied by the caller, or 'http' as the safe default.
        # Using re.sub + unconditional prepend avoids the bug where a template
        # already starting with 'http://' silently discards a caller-supplied 'https'.
        composed = re.sub(r'^https?://', '', composed)  # strip any baked-in protocol
        composed = f"{protocol}://{composed}"          # prepend the correct one

        logger.debug(f"Composed URL: {composed}")
        return composed

    @staticmethod
    def jsonify(data)-> json:
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
            raise ValueError(f"Invalid JSON data: {str(e)}")

    @staticmethod
    def convert64(img: str) -> bytes:
        """Convert an image file to base64 string

        Args:
            img: path to image file

        Returns:
            Base64 encoded bytes of the image

        Raises:
            ValueError: If img is not a valid string or file doesn't exist
            Exception: For other file reading errors
        """
        logger = logging.getLogger(__name__)

        if not isinstance(img, str) or not img.strip():
            raise ValueError("Image path must be a non-empty string")

        # Additional security checks to prevent path traversal attacks
        img = img.strip()

        # Check for path traversal attempts using multiple methods
        # 1. Check for forbidden patterns like .. in the path
        if '..' in img:
            # More comprehensive check for directory traversal
            if img == '..' or img.startswith('../') or img.endswith('/..') or '/..' in img:
                raise ValueError("Image path contains invalid traversal characters")

            # Also check for encoded versions of path traversal
            if '%2e%2e' in img.lower():
                raise ValueError("Image path contains encoded traversal characters")

        # 2. Check for symbolic links to prevent directory traversal
        if os.path.islink(img):
            raise ValueError("Image path must not be a symbolic link")

        # 3. Check that the file exists and is actually a file (not a directory)
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")

        if not os.path.isfile(img):
            raise ValueError("Image path must point to a file, not a directory")

        try:
            logger.debug(f"Converting image to base64: {img}")
            result = ''
            with open(img, "rb") as image_file:
                result = base64.b64encode(image_file.read())
            logger.debug("Image conversion completed successfully")
            return result
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}", exc_info=True)
            raise Exception(f"Error reading image file {img}: {str(e)}")

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
            except Exception:
                pass  # Ignore cleanup errors

        # Clean up httpx.AsyncClient if provided
        if client is not None:
            try:
                # For sync methods, we can close directly
                if hasattr(client, 'close'):
                    client.close()
                # For async methods, we need to await aclose()
                elif hasattr(client, 'aclose'):
                    import asyncio
                    # This is a bit tricky - for async context we'd want to await,
                    # but this is a sync utility, so we'll just ignore it here
                    pass
            except Exception:
                pass  # Ignore cleanup errors