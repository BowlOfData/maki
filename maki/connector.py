import requests

from . import utils

class Connector:

    def simple(url: str, data: dict)-> dict:
        response = requests.post(url, json = data)
        jsonify = utils.Utils.jsonify(response.text)
        return jsonify["response"]