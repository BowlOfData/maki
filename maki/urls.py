from enum import Enum

class Actions(Enum):
    GENERATE = 'generate'
    VERSION = 'version'

GENERIC_LLAMA_URL = "{domain}:{port}/api/{action}"