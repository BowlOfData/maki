import requests
import json
import logging
from .utils import Utils

class Connector:

    @staticmethod
    def simple(url: str, prompt: dict) -> str:
        """Send a simple request to the Ollama API

        Args:
            url: the API endpoint URL
            prompt: the data to send

        Returns:
            The LLM response text as a string

        Raises:
            ValueError: If url or prompt is invalid
            requests.RequestException: For HTTP request errors
            json.JSONDecodeError: For JSON parsing errors
            Exception: For other errors
        """
        logger = logging.getLogger(__name__)

        logger.debug(f"Preparing to send simple request to URL: {url}")

        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        if not isinstance(prompt, dict):
            raise ValueError("Prompt must be a dictionary")

        logger.debug(f"Sending simple request to URL: {url}")
        logger.debug(f"Request data: {prompt}")

        try:
            response = requests.post(url, json=prompt, timeout=180)
            response.raise_for_status()  # Raise an exception for bad status codes
            logger.debug("HTTP request completed successfully")
            jsonify = Utils.jsonify(response.text)
            # Check if response contains the expected structure
            if "response" not in jsonify:
                raise Exception("Invalid API response format: missing 'response' field")
            logger.debug("Request completed successfully")
            return jsonify["response"]
        except requests.exceptions.Timeout:
            logger.error("HTTP request timed out")
            raise Exception("HTTP request timed out")
        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise Exception(f"HTTP request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise Exception(f"JSON parsing failed: {str(e)}")
        except KeyError as e:
            logger.error(f"API response structure error: {str(e)}")
            raise Exception(f"API response structure error: {str(e)}")

    @staticmethod
    def version(url: str) -> str:
        """Get version information from the Ollama API

        Args:
            url: the version API endpoint URL

        Returns:
            The version information as text

        Raises:
            ValueError: If url is invalid
            requests.RequestException: For HTTP request errors
            Exception: For other errors
        """
        logger = logging.getLogger(__name__)

        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        logger.debug(f"Fetching version from URL: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.debug("Version request completed successfully")
            return response.text
        except requests.exceptions.Timeout:
            logger.error("HTTP request timed out")
            raise Exception("HTTP request timed out")
        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise Exception(f"HTTP request failed: {str(e)}")
    
