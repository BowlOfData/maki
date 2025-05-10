import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from . import utils
from . import connector

class Maki:

    @classmethod
    def __init__(self, url: str, port: str, model: str):
        self.url = url
        self.port = port
        self.model = model

    @classmethod
    def request(self, data: dict) -> str:
        url = utils.Utils.compose_url(self.url, self.port)
        result = connector.Connector.simple(url, data)
        return result
    
    def get_model(self)->str:
        return self.model