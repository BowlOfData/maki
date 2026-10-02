"""
Central configuration defaults for the Maki project.
"""

from __future__ import annotations

import logging
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_OLLAMA_SCHEME = os.getenv("MAKI_OLLAMA_SCHEME", "http")
DEFAULT_OLLAMA_HOST = os.getenv("MAKI_OLLAMA_HOST", "localhost")
DEFAULT_OLLAMA_PORT = os.getenv("MAKI_OLLAMA_PORT", "11434")
DEFAULT_OLLAMA_BASE_URL = os.getenv(
    "MAKI_OLLAMA_BASE_URL",
    f"{DEFAULT_OLLAMA_SCHEME}://{DEFAULT_OLLAMA_HOST}:{DEFAULT_OLLAMA_PORT}",
)

DEFAULT_MODEL = os.getenv("MAKI_DEFAULT_MODEL", "gemma3")
DEFAULT_TEMPERATURE = _get_float("MAKI_DEFAULT_TEMPERATURE", 0.7)
DEFAULT_REQUEST_TIMEOUT = _get_int("MAKI_REQUEST_TIMEOUT", 120)
DEFAULT_HTTP_TIMEOUT = _get_int("MAKI_HTTP_TIMEOUT", 10)
DEFAULT_HTTP_READ_TIMEOUT = _get_int("MAKI_HTTP_READ_TIMEOUT", 60)

DEFAULT_LOG_LEVEL_NAME = os.getenv("MAKI_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_LEVEL = getattr(logging, DEFAULT_LOG_LEVEL_NAME, logging.INFO)
DEFAULT_LOG_FORMAT = os.getenv(
    "MAKI_LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_WEB_USER_AGENT = os.getenv(
    "MAKI_WEB_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
DEFAULT_BROWSER_ACCEPT_LANGUAGE = os.getenv(
    "MAKI_ACCEPT_LANGUAGE",
    "en-US,en;q=0.9",
)
PLUGIN_PACKAGE_PREFIX = "maki.plugins"
PLUGIN_REQUIRED_FILES = ("__init__.py", "README.md", "example_usage.py")

DEFAULT_OCR_OUTPUT_DIR = os.path.expanduser(
    os.getenv("MAKI_OCR_OUTPUT_DIR", "~/maki_ocr_output")
)
DEFAULT_OCR_MODEL = os.getenv("MAKI_OCR_MODEL", "glm-ocr")

OPENAI_API_KEY_ENV      = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL    = os.getenv("MAKI_OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY_ENV   = "ANTHROPIC_API_KEY"
DEFAULT_ANTHROPIC_MODEL = os.getenv("MAKI_ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENROUTER_API_KEY_ENV      = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_MODEL    = os.getenv("MAKI_OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_BASE_URL = os.getenv("MAKI_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL_ENV     = "OPENROUTER_SITE_URL"
OPENROUTER_APP_NAME_ENV     = "OPENROUTER_APP_NAME"

LLAMACPP_API_KEY_ENV      = "LLAMACPP_API_KEY"
DEFAULT_LLAMACPP_MODEL    = os.getenv("MAKI_LLAMACPP_MODEL", "local-model")
DEFAULT_LLAMACPP_BASE_URL = os.getenv("MAKI_LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1")

__all__ = [
    "DEFAULT_OLLAMA_SCHEME",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_PORT",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_HTTP_READ_TIMEOUT",
    "DEFAULT_LOG_LEVEL_NAME",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_WEB_USER_AGENT",
    "DEFAULT_BROWSER_ACCEPT_LANGUAGE",
    "PLUGIN_PACKAGE_PREFIX",
    "PLUGIN_REQUIRED_FILES",
    "DEFAULT_OCR_OUTPUT_DIR",
    "DEFAULT_OCR_MODEL",
    "OPENAI_API_KEY_ENV",
    "DEFAULT_OPENAI_MODEL",
    "ANTHROPIC_API_KEY_ENV",
    "DEFAULT_ANTHROPIC_MODEL",
    "OPENROUTER_API_KEY_ENV",
    "DEFAULT_OPENROUTER_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "OPENROUTER_SITE_URL_ENV",
    "OPENROUTER_APP_NAME_ENV",
    "LLAMACPP_API_KEY_ENV",
    "DEFAULT_LLAMACPP_MODEL",
    "DEFAULT_LLAMACPP_BASE_URL",
]
