import json
import base64
import os
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
        """
        return GENERIC_LLAMA_URL.format(domain=url, port=port, action=action)

    @staticmethod
    def jsonify(data)-> json:
        """Parse JSON data

        Args:
            data: JSON string to parse

        Returns:
            Parsed JSON object
        """
        return json.loads(data)

    @staticmethod
    def convert64(img: str)-> str:
        """Convert an image file to base64 string

        Args:
            img: path to image file

        Returns:
            Base64 encoded string of the image

        Raises:
            FileNotFoundError: If the image file doesn't exist
            Exception: For other file reading errors
        """
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")

        result = ''
        with open(img, "rb") as image_file:
            result = base64.b64encode(image_file.read())

        return result