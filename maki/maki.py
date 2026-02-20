from maki.utils import Utils
from maki.connector import Connector
from maki.llm_objects.ollama_payload import OLLAMA_PAYLOAD
from maki.urls import Actions

class Maki:
    def __init__(self, url: str, port: str, model: str, temperature=0):
        """ Initialize the Maki object

        Args:
            url: the Ollama url
            port: the Ollama port
            model: the model to use
            temperature: the LLM temperature
        """
        self.url = url
        self.port = port
        self.model = model
        self.temperature = temperature

    def request(self, prompt: str) -> str:
        """ Send the request to the LLM

        Args:
            prompt: user prompt

        returns a string containing the payload
        """
        url = Utils.compose_url(self.url, self.port, Actions.GENERATE)
        data = self._compose_data(prompt)
        result = Connector.simple(url, data)
        return result

    def version(self) -> str:
        """ Returns the LLM version

        returns a string containing the version
        """
        url = Utils.compose_url(self.url, self.port, Actions.VERSION)
        result = Connector.version(url)
        return result

    def _compose_data(self, prompt:str, imgs=None) -> str:
        OLLAMA_PAYLOAD["model"] = self._get_model()
        OLLAMA_PAYLOAD["prompt"] = prompt

        if(self._get_temperature()):
            OLLAMA_PAYLOAD["options"] = {"temperature":self._get_temperature()}

        if(imgs):
            OLLAMA_PAYLOAD["images"] = imgs

        return OLLAMA_PAYLOAD

    def request_with_images(self, prompt: str, img:str)-> str:
        """ Send a request with image input to the LLM

        Args:
            prompt: user prompt
            img: path to image file

        returns a string containing the response
        """
        url = Utils.compose_url(self.url, self.port, Actions.GENERATE)
        converted_imgs = Utils.convert64(img)
        imgs = []
        imgs.append(converted_imgs.decode("utf-8"))
        data = self._compose_data(prompt, imgs=imgs)
        result = Connector.simple(url, data)
        return result

    def _get_model(self)->str:
        return self.model

    def _get_temperature(self)->int:
        return self.temperature