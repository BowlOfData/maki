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
            requests.RequestException: For HTTP request errors
            json.JSONDecodeError: For JSON parsing errors
        """
        try:
            response = requests.post(url, json=prompt)
            response.raise_for_status()  # Raise an exception for bad status codes
            jsonify = Utils.jsonify(response.text)
            return jsonify["response"]
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing failed: {str(e)}")

    @staticmethod
    def version(url: str)-> dict:
        """Get version information from the Ollama API

        Args:
            url: the version API endpoint URL

        Returns:
            The version information as text

        Raises:
            requests.RequestException: For HTTP request errors
        """
        try:
            response = requests.post(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
    
