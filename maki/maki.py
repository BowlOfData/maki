import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .utils import Utils
from .connector import Connector
from maki.llm_objects.ollama_payload import OLLAMA_PAYLOAD

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
        url = Utils.compose_url(self.url, self.port)
        result = Connector.simple(url, self.compose_data(prompt))
        return result
    
    @classmethod
    def _compose_data(self, prompt:str) -> str:
        OLLAMA_PAYLOAD["model"] = self.get_model()
        OLLAMA_PAYLOAD["prompt"] = prompt

        if(self.get_temperature()):
            OLLAMA_PAYLOAD["options"] = {"temperature":self.get_temperature()}
        
        return OLLAMA_PAYLOAD

    @classmethod
    def _get_model(self)->str:
        return self.model
    
    @classmethod
    def _get_temperature(self)->int:
        return self.teperature