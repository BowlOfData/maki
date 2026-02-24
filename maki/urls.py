from enum import StrEnum

class Actions(StrEnum):
    GENERATE = 'generate'
    VERSION = 'version'

GENERIC_LLAMA_URL = "{domain}:{port}/api/{action}"