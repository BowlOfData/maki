import requests

from .utils import Utils

class Connector:

    @classmethod
    def simple(self, url: str, prompt: dict)-> dict:
        response = requests.post(url, json = prompt)
        jsonify = Utils.jsonify(response.text)
        return jsonify["response"]
    
    @classmethod
    def version(self, url: str)-> dict:
        response = requests.post(url)
        #jsonify = Utils.jsonify(response.text)
        return response.text
    
