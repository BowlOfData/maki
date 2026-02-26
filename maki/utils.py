import json
import base64
import os
import logging
import re
import ipaddress
from .urls import GENERIC_LLAMA_URL

class Utils:

    # List of private IP ranges that should be blocked to prevent SSRF
    PRIVATE_IP_RANGES = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "::1/128",
        "fe80::/10"
    ]

    # Blacklisted domains that should be blocked
    BLACKLISTED_DOMAINS = [
        "localhost",
        "127.0.0.1",
        "::1"
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

        # Check for private IP addresses
        try:
            # Try to parse as IP address
            ip = ipaddress.ip_address(domain)
            # Check if it's a private IP
            for ip_range in Utils.PRIVATE_IP_RANGES:
                if ip in ipaddress.ip_network(ip_range):
                    raise ValueError(f"Access to private IP address '{domain}' is not allowed")
        except ValueError:
            # Not an IP address, check as domain name
            # Basic domain validation
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', domain):
                raise ValueError(f"Invalid domain format: {domain}")

            # Check for potentially malicious patterns
            if '..' in domain:
                raise ValueError("Domain contains invalid pattern")

            # Check for special characters that could be used in SSRF attacks
            if re.search(r'[^\w\.\-]', domain):
                raise ValueError("Domain contains invalid characters")

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
        Utils._validate_domain(url)
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