import json
from .urls import GENERIC_LLAMA_URL

class Utils:

    def compose_url(url: str, port: str, action: str) -> str:

        return GENERIC_LLAMA_URL.format(domain=url,port=port,action=action)
    
    def jsonify(data)-> json:
        return json.loads(data)