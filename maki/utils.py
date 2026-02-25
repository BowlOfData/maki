import json
import base64
import os
import logging
from .urls import GENERIC_LLAMA_URL

class Utils:

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

        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        if not isinstance(port, str) or not port.strip():
            raise ValueError("Port must be a non-empty string")

        if not isinstance(action, str) or not action.strip():
            raise ValueError("Action must be a non-empty string")

        composed = GENERIC_LLAMA_URL.format(domain=url.strip(), port=port, action=action.strip())
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