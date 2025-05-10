from utils import Utils

class Maki:

    @classmethod
    def __init__(self, url: str, port: str, model: str):
        self.url = url
        self.port = port
        self.model = model

    @classmethod
    def request(self) -> str:
        url = Utils.compose_url(self.url, self.port)
        
        return ""