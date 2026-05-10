"""
Verification tests for shared configuration defaults.
"""

from unittest.mock import patch

from maki import config
from maki.maki import Maki
from maki.makiLLama import MakiLLama


def test_config_exports_expected_defaults():
    assert config.DEFAULT_OLLAMA_HOST
    assert config.DEFAULT_OLLAMA_PORT
    assert config.DEFAULT_OLLAMA_BASE_URL.startswith("http")
    assert isinstance(config.DEFAULT_TEMPERATURE, float)


def test_maki_uses_shared_defaults():
    client = Maki()

    assert client.url == config.DEFAULT_OLLAMA_HOST
    assert client.port == config.DEFAULT_OLLAMA_PORT
    assert client.model == config.DEFAULT_MODEL
    assert client.temperature == config.DEFAULT_TEMPERATURE


def test_makillama_uses_shared_defaults():
    with patch.object(MakiLLama, "_verify_connection", return_value=None):
        client = MakiLLama()

    assert client.base_url == config.DEFAULT_OLLAMA_BASE_URL
    assert client.model == config.DEFAULT_MODEL
    assert client.temperature == config.DEFAULT_TEMPERATURE
    assert client.timeout == config.DEFAULT_REQUEST_TIMEOUT
