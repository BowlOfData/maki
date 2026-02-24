import requests
import json
from .utils import Utils

class Connector:

    @staticmethod
    def simple(url: str, prompt: dict)-> dict:
        """Send a simple request to the Ollama API

        Args:
            url: the API endpoint URL
            prompt: the data to send

        Returns:
            The parsed response from the API

        Raises:
            ValueError: If url or prompt is invalid
            requests.RequestException: For HTTP request errors
            json.JSONDecodeError: For JSON parsing errors
            Exception: For other errors
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        if not isinstance(prompt, dict):
            raise ValueError("Prompt must be a dictionary")

        try:
            response = requests.post(url, json=prompt, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            jsonify = Utils.jsonify(response.text)
            # Check if response contains the expected structure
            if "response" not in jsonify:
                raise Exception("Invalid API response format: missing 'response' field")
            return jsonify["response"]
        except requests.exceptions.Timeout:
            raise Exception("HTTP request timed out")
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"API response structure error: {str(e)}")

    @staticmethod
    def version(url: str)-> dict:
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
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        try:
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            raise Exception("HTTP request timed out")
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
    
