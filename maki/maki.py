from .utils import Utils
from .connector import Connector
from .urls import Actions
import re

class Maki:
    def __init__(self, url: str, port: str, model: str, temperature=0):
        """ Initialize the Maki object

        Args:
            url: the Ollama url
            port: the Ollama port
            model: the model to use
            temperature: the LLM temperature

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate URL
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        # Validate port
        if not isinstance(port, str):
            raise ValueError("Port must be a string")

        if not re.match(r'^[0-9]+$', port):
            raise ValueError("Port must be a valid port number (numeric string)")

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

    def request(self, prompt: str) -> str:
        """ Send the request to the LLM

        Args:
            prompt: user prompt

        Returns:
            A string containing the payload

        Raises:
            ValueError: If prompt is not a valid string
            Exception: For HTTP request or JSON parsing errors
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        url = Utils.compose_url(self.url, self.port, Actions.GENERATE)
        data = self._compose_data(prompt)
        result = Connector.simple(url, data)
        return result

    def version(self) -> str:
        """ Returns the LLM version

        Returns:
            A string containing the version

        Raises:
            Exception: For HTTP request errors
        """
        url = Utils.compose_url(self.url, self.port, Actions.VERSION)
        result = Connector.version(url)
        return result

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

        # Create a new payload dict for each request (thread-safe)
        payload = {
            "model": self._get_model(),
            "prompt": prompt.strip(),
            "stream": False
        }

        if self._get_temperature() is not None:
            payload["options"] = {"temperature": self._get_temperature()}

        if imgs:
            # Validate that imgs is a list
            if not isinstance(imgs, list):
                raise ValueError("Images must be provided as a list")
            payload["images"] = imgs

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
            Exception: For HTTP request or file reading errors
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        if not isinstance(img, str) or not img.strip():
            raise ValueError("Image path must be a non-empty string")

        url = Utils.compose_url(self.url, self.port, Actions.GENERATE)
        converted_imgs = Utils.convert64(img)
        imgs = []
        imgs.append(converted_imgs.decode("utf-8"))
        data = self._compose_data(prompt, imgs=imgs)
        result = Connector.simple(url, data)
        return result

    def _get_model(self)->str:
        return self.model

    def _get_temperature(self) -> float:
        return self.temperature