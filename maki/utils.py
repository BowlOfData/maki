import json
import base64
from .urls import GENERIC_LLAMA_URL

class Utils:

    def compose_url(url: str, port: str, action: str) -> str:

        return GENERIC_LLAMA_URL.format(domain=url,port=port,action=action)
    
    def jsonify(data)-> json:
        return json.loads(data)
    
    def convert64(img: str)-> str:
        
        result = ''
        with open(img, "rb") as image_file:
            result = base64.b64encode(image_file.read())

        return result