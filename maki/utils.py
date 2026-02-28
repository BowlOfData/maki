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
        "127.0.0.0/8",
        "::1/128",
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
        "2001:db8::/32",   # Documentation examples
        "fc00::/7",        # Unique local addresses
    ]

    # Blacklisted domains that should be blocked
    BLACKLISTED_DOMAINS = [
        "0.0.0.0",
        "255.255.255.255"
    ]

    # Additional blacklisted patterns
    BLACKLISTED_PATTERNS = [
        r"^\d+\.\d+\.\d+\.\d+$",  # Plain IP addresses
        r"^[0-9a-fA-F:]+$",       # IPv6 addresses
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

        # Check if it's an IP address
        try:
            # Try to parse as IP address
            ip = ipaddress.ip_address(domain)
            # Check if it's a private IP
            for ip_range in Utils.PRIVATE_IP_RANGES:
                if ip in ipaddress.ip_network(ip_range):
                    raise ValueError(f"Access to private IP address '{domain}' is not allowed")
            # Additional check for specific IP ranges that are dangerous
            if ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError(f"Access to special IP address '{domain}' is not allowed")
        except ValueError:
            # Not an IP address, check as domain name
            # Basic domain validation
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', domain):
                raise ValueError(f"Invalid domain format: {domain}")

            # Check for potentially malicious patterns
            if '..' in domain:
                raise ValueError("Domain contains invalid pattern")

            # Check for special characters that could be used in SSRF attacks
            if re.search(r'[^\w\.\-\:]', domain):
                raise ValueError("Domain contains invalid characters")

            # Check for blacklisted patterns
            for pattern in Utils.BLACKLISTED_PATTERNS:
                if re.match(pattern, domain):
                    raise ValueError(f"Domain '{domain}' matches blacklisted pattern")

            # Additional domain validation
            if len(domain) > 253:
                raise ValueError("Domain name too long")

            # Check for valid label lengths (each part between dots)
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
                Utils._validate_domain(domain_part)
        else:
            Utils._validate_domain(original_url)

        Utils._validate_port(port)

        if not isinstance(action, str) or not action.strip():
            raise ValueError("Action must be a non-empty string")

        # Sanitize inputs
        url = url.strip()
        port = port.strip()
        action = action.strip()

        # Additional URL sanitization
        # Remove any dangerous characters that could be used in URL manipulation
        url = re.sub(r'[^a-zA-Z0-9.\-:]', '', url)
        action = re.sub(r'[^a-zA-Z0-9\-_/.]', '', action)

        # Validate that we don't have any protocol in the URL (to prevent SSRF)
        # But allow localhost and valid local domains
        if url.startswith(('http://', 'https://')):
            # Extract domain for validation to allow localhost and valid local domains
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            domain = parsed.hostname or parsed.netloc
            if domain and domain not in ['localhost', '127.0.0.1', '::1']:
                # For non-local domains, validate against SSRF rules
                try:
                    ip = ipaddress.ip_address(domain)
                    # Allow localhost and loopback addresses, but validate other IPs
                    if not (ip.is_loopback or ip.is_link_local):
                        # Check if it's a private IP that should be blocked
                        for ip_range in Utils.PRIVATE_IP_RANGES:
                            if ip in ipaddress.ip_network(ip_range):
                                raise ValueError(f"Access to private IP address '{domain}' is not allowed")
                except ValueError:
                    # If it's not an IP address, validate as domain
                    pass
        else:
            # No protocol - validate as domain or IP
            pass

        # Ensure we have a valid domain format after sanitization
        if not re.match(r'^[a-zA-Z0-9.\-:]+$', url):
            raise ValueError("Invalid domain format after sanitization")

        # Additional security check to prevent directory traversal in the action
        if '..' in action:
            raise ValueError("Action contains invalid characters")

        composed = GENERIC_LLAMA_URL.format(domain=url, port=port, action=action)
        # Add http:// protocol if not present
        if not composed.startswith(('http://', 'https://')):
            composed = f"http://{composed}"

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
    def convert64(img: str)-> str:
        """Convert an image file to base64 string

        Args:
            img: path to image file

        Returns:
            Base64 encoded string of the image

        Raises:
            ValueError: If img is not a valid string or file doesn't exist
            Exception: For other file reading errors
        """
        logger = logging.getLogger(__name__)

        if not isinstance(img, str) or not img.strip():
            raise ValueError("Image path must be a non-empty string")

        # Additional security checks to prevent path traversal attacks
        img = img.strip()

        # Check for path traversal attempts (but allow absolute paths)
        if img.startswith('../') or '/../' in img:
            raise ValueError("Image path contains invalid characters")

        # Check for symbolic links to prevent directory traversal
        if os.path.islink(img):
            raise ValueError("Image path must not be a symbolic link")

        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")

        try:
            logger.debug(f"Converting image to base64: {img}")
            result = ''
            with open(img, "rb") as image_file:
                result = base64.b64encode(image_file.read())
            logger.debug("Image conversion completed successfully")
            return result
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            raise Exception(f"Error reading image file {img}: {str(e)}")