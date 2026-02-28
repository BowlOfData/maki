from typing import Union
from .utils import Utils
from .connector import Connector
from .urls import Actions
from .exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError
import re
import logging

class Maki:
    def __init__(self, url: str, port: Union[str, int], model: str, temperature=0):
        """ Initialize the Maki object

        Args:
            url: the Ollama url
            port: the Ollama port
            model: the model to use
            temperature: the LLM temperature

        Raises:
            ValueError: If any parameter is invalid
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Validate URL
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        # Validate port
        if not isinstance(port, (str, int)):
            raise ValueError("Port must be a string or integer")

        if isinstance(port, str) and not re.match(r'^[0-9]+$', port):
            raise ValueError("Port must be a valid port number (numeric string)")

        if isinstance(port, int):
            port = str(port)

        # Validate model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string")

        # Validate temperature
        if not isinstance(temperature, (int, float)):
            raise ValueError("Temperature must be a number")

        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")

        self.url = url.strip()
        self.port = port
        self.model = model.strip()
        self.temperature = float(temperature)

        self.logger.info(f"Maki initialized with URL: {self.url}, Port: {self.port}, Model: {self.model}")

    def request(self, prompt: str) -> str:
        """ Send the request to the LLM

        Args:
            prompt: user prompt

        Returns:
            A string containing the payload

        Raises:
            ValueError: If prompt is not a valid string
            MakiNetworkError: For network-related errors
            MakiTimeoutError: For timeout errors
            MakiAPIError: For API response errors
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        self.logger.debug(f"Sending request with prompt: {prompt[:100]}...")
        url = Utils.compose_url(self.url, self.port, Actions.GENERATE.value)
        data = self._compose_data(prompt)
        try:
            result = Connector.simple(url, data)
            self.logger.debug("Request completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            # Re-raise with more specific type if needed
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            else:
                raise MakiNetworkError(f"HTTP request failed: {str(e)}")

    def version(self) -> str:
        """ Returns the LLM version

        Returns:
            A string containing the version

        Raises:
            MakiNetworkError: For network-related errors
            MakiTimeoutError: For timeout errors
            MakiAPIError: For API response errors
        """
        self.logger.debug("Fetching version information")
        url = Utils.compose_url(self.url, self.port, Actions.VERSION.value)
        try:
            result = Connector.version(url)
            self.logger.debug("Version information retrieved successfully")
            return result
        except Exception as e:
            self.logger.error(f"Failed to retrieve version: {str(e)}")
            # Re-raise with more specific type if needed
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            else:
                raise MakiNetworkError(f"Failed to retrieve version: {str(e)}")

    def _compose_data(self, prompt:str, imgs=None) -> dict:
        """Compose the data payload for the LLM request

        Args:
            prompt: The prompt to send
            imgs: Optional list of image data

        Returns:
            The composed data payload as a dictionary

        Raises:
            ValueError: If prompt is not valid
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        self.logger.debug("Composing data payload")
        # Create a new payload dict for each request (thread-safe)
        payload = {
            "model": self._get_model(),
            "prompt": prompt.strip(),
            "stream": False
        }

        temperature = self._get_temperature()
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        if imgs:
            # Validate that imgs is a list
            if not isinstance(imgs, list):
                raise ValueError("Images must be provided as a list")
            payload["images"] = imgs

        self.logger.debug("Data payload composed successfully")
        return payload

    def request_with_images(self, prompt: str, img:str)-> str:
        """ Send a request with image input to the LLM

        Args:
            prompt: user prompt
            img: path to image file

        Returns:
            A string containing the response

        Raises:
            ValueError: If prompt or img is not valid
            MakiNetworkError: For network-related errors
            MakiTimeoutError: For timeout errors
            MakiAPIError: For API response errors
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        if not isinstance(img, str) or not img.strip():
            raise ValueError("Image path must be a non-empty string")

        self.logger.debug(f"Sending request with image: {img}")
        url = Utils.compose_url(self.url, self.port, Actions.GENERATE.value)
        try:
            converted_imgs = Utils.convert64(img)
            imgs = [converted_imgs.decode("utf-8")]
            data = self._compose_data(prompt, imgs=imgs)
            result = Connector.simple(url, data)
            self.logger.debug("Request with image completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Request with image failed: {str(e)}")
            # Re-raise with more specific type if needed
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            else:
                raise MakiNetworkError(f"HTTP request with image failed: {str(e)}")

    def _get_model(self)->str:
        return self.model

    def _get_temperature(self) -> float:
        return self.temperature