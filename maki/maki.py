from maki.utils import Utils
from maki.connector import Connector
from maki.llm_objects.ollama_payload import OLLAMA_PAYLOAD
from maki.urls import Actions

class Maki:

    @classmethod
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
        self.teperature = temperature

    @classmethod
    def request(self, prompt: str) -> str:
        """ Send the request to the LLM

        Args:
            prompt: user prompt

        returns a string containing the payload
        """
        url = Utils.compose_url(self.url, self.port, Actions.GENERATE)
        result = Connector.simple(url, self._compose_data(prompt))
        return result
    
    @classmethod
    def version(self) -> str:

        """ Returns the LLM version

        returns a string containing the version
        """

        url = Utils.compose_url(self.url, self.port, Actions.VERSION)
        result = Connector.version(url)
        return result

    @classmethod
    def _compose_data(self, prompt:str) -> str:
        OLLAMA_PAYLOAD["model"] = self._get_model()
        OLLAMA_PAYLOAD["prompt"] = prompt

        if(self._get_temperature()):
            OLLAMA_PAYLOAD["options"] = {"temperature":self._get_temperature()}
        
        return OLLAMA_PAYLOAD

    @classmethod
    def _get_model(self)->str:
        return self.model
    
    @classmethod
    def _get_temperature(self)->int:
        return self.teperature