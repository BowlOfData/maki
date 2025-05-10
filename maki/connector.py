import requests

class Connector:

    def simple(url: str):
        requests.request("POST", url)
        pass