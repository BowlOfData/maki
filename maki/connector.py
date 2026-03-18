import urllib.parse
import requests
import logging
from .utils import Utils
from .exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError, MakiValidationError

logger = logging.getLogger(__name__)


class Connector:

    @staticmethod
    def _validate_url(url: str) -> None:
        """Validate URL format and apply SSRF protection.

        Args:
            url: The URL to validate

        Raises:
            MakiValidationError: If the URL is malformed, uses a disallowed
                scheme, or targets a private/reserved address.
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.scheme not in ('http', 'https'):
            raise MakiValidationError("URL must use http or https protocol")
        if parsed.hostname:
            try:
                Utils._validate_domain(parsed.hostname)
            except ValueError as e:
                raise MakiValidationError(str(e)) from e

    @staticmethod
    def _raise_for_http_error(exc: requests.exceptions.HTTPError) -> None:
        """Translate a requests.HTTPError into the appropriate Maki exception.

        Raises:
            MakiNetworkError: For 5xx or unknown-status errors.
            MakiAPIError: For 4xx errors.
        """
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code is not None and status_code >= 500:
            raise MakiNetworkError(f"HTTP server error {status_code}: {exc}") from exc
        elif status_code is not None:
            raise MakiAPIError(f"HTTP client error {status_code}: {exc}") from exc
        else:
            raise MakiNetworkError(f"HTTP request failed (unknown status): {exc}") from exc

    @staticmethod
    def simple(url: str, prompt: dict) -> dict:
        """Send a simple request to the Ollama API

        Args:
            url: the API endpoint URL
            prompt: the data to send

        Returns:
            The full parsed JSON response from the API as a dict.

        Raises:
            MakiValidationError: If url or prompt is invalid
            MakiTimeoutError: For HTTP timeout errors
            MakiNetworkError: For other HTTP request errors
            MakiAPIError: For API response errors
        """
        if not isinstance(url, str) or not url.strip():
            raise MakiValidationError("URL must be a non-empty string")

        if not isinstance(prompt, dict):
            raise MakiValidationError("Prompt must be a dictionary")

        Connector._validate_url(url)

        logger.debug(f"Sending simple request to URL: {url}")
        logger.debug(f"Request data: {prompt}")

        try:
            response = requests.post(url, json=prompt, timeout=180)
            response.raise_for_status()
            logger.debug("HTTP request completed successfully")
            jsonify = Utils.jsonify(response.text)
            if not isinstance(jsonify, dict):
                raise MakiAPIError("Invalid API response format: expected a JSON object")
            logger.debug("Request completed successfully")
            return jsonify
        except requests.exceptions.Timeout as e:
            logger.error("HTTP request timed out", exc_info=True)
            raise MakiTimeoutError("HTTP request timed out") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"HTTP connection failed: {str(e)}", exc_info=True)
            raise MakiNetworkError(f"HTTP connection failed: {str(e)}") from e
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP request failed: {str(e)}", exc_info=True)
            Connector._raise_for_http_error(e)
        except (MakiAPIError, MakiNetworkError, MakiTimeoutError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in HTTP request: {str(e)}", exc_info=True)
            raise MakiNetworkError(f"Unexpected error in HTTP request: {str(e)}") from e

    @staticmethod
    def version(url: str) -> str:
        """Get version information from the Ollama API

        Args:
            url: the version API endpoint URL

        Returns:
            The version information as text

        Raises:
            MakiValidationError: If url is invalid
            MakiTimeoutError: For HTTP timeout errors
            MakiNetworkError: For other HTTP request errors
        """
        if not isinstance(url, str) or not url.strip():
            raise MakiValidationError("URL must be a non-empty string")

        Connector._validate_url(url)

        logger.debug(f"Fetching version from URL: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.debug("Version request completed successfully")
            return response.text
        except requests.exceptions.Timeout as e:
            logger.error("HTTP request timed out", exc_info=True)
            raise MakiTimeoutError("HTTP request timed out") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"HTTP connection failed: {str(e)}", exc_info=True)
            raise MakiNetworkError(f"HTTP connection failed: {str(e)}") from e
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP request failed: {str(e)}", exc_info=True)
            Connector._raise_for_http_error(e)
        except Exception as e:
            logger.error(f"Unexpected error in version request: {str(e)}", exc_info=True)
            raise MakiNetworkError(f"Unexpected error in version request: {str(e)}") from e
