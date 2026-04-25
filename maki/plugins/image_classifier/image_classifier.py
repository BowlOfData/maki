"""
Image classifier plugin for Maki agents.

Classifies images using a vision-capable Ollama model (e.g. gemma4:26b).
The prompt and system prompt are provided by the caller, keeping this plugin
domain-agnostic and reusable for any classification task.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = ["classify_image", "classify_image_async"]


class ImageClassifier:
    """Generic image classifier backed by a vision-capable Ollama model."""

    ALLOWED_METHODS = _ALLOWED_METHODS

    def __init__(self, maki_instance=None) -> None:
        self._maki = maki_instance

    def _encode(self, image_path: str) -> str:
        return base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def classify_image(
        self,
        image_path: str,
        prompt: str,
        system: Optional[str] = None,
        valid_labels: Optional[List[str]] = None,
        fallback_label: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Classify a single image (synchronous).

        Args:
            image_path:    Path to the image file.
            prompt:        User-facing question / instruction for the model.
            system:        Optional system prompt that frames the task.
            valid_labels:  If provided, any response not in this list is replaced
                           with fallback_label.
            fallback_label: Label to use when the model response is not recognized.

        Returns:
            Dict with keys: success (bool), label (str), image_path (str), error (str|None)
        """
        if self._maki is None:
            return {"success": False, "image_path": image_path, "error": "No maki instance configured"}

        try:
            image_b64 = self._encode(image_path)
            response = self._maki.chat_with_image(prompt, image_b64=image_b64, system=system)
            label = response.content.strip().lower()
            if valid_labels is not None and label not in valid_labels:
                label = fallback_label
            logger.debug("Classified %s → %s", image_path, label)
            return {"success": True, "label": label, "image_path": image_path, "error": None}
        except Exception as e:
            logger.warning("Failed to classify %s: %s", image_path, e)
            return {"success": False, "label": fallback_label, "image_path": image_path, "error": str(e)}

    def classify_image_async(
        self,
        image_path: str,
        prompt: str,
        system: Optional[str] = None,
        valid_labels: Optional[List[str]] = None,
        fallback_label: str = "unknown",
    ) -> Dict[str, Any]:
        """Sync wrapper around the async coroutine — for use from non-async code."""
        return asyncio.get_event_loop().run_until_complete(
            self._classify_async(image_path, prompt, system, valid_labels, fallback_label)
        )

    async def classify_image_async_coro(
        self,
        image_path: str,
        prompt: str,
        system: Optional[str] = None,
        valid_labels: Optional[List[str]] = None,
        fallback_label: str = "unknown",
    ) -> Dict[str, Any]:
        """Async coroutine — await this directly inside an asyncio event loop."""
        return await self._classify_async(image_path, prompt, system, valid_labels, fallback_label)

    async def _classify_async(
        self,
        image_path: str,
        prompt: str,
        system: Optional[str],
        valid_labels: Optional[List[str]],
        fallback_label: str,
    ) -> Dict[str, Any]:
        if self._maki is None:
            return {"success": False, "image_path": image_path, "error": "No maki instance configured"}
        try:
            image_b64 = await asyncio.to_thread(self._encode, image_path)
            response = await self._maki.async_chat(prompt, images=[image_b64], system=system)
            label = response.content.strip().lower()
            if valid_labels is not None and label not in valid_labels:
                label = fallback_label
            logger.debug("Classified %s → %s", image_path, label)
            return {"success": True, "label": label, "image_path": image_path, "error": None}
        except Exception as e:
            logger.warning("Failed to classify %s: %s", image_path, e)
            return {"success": False, "label": fallback_label, "image_path": image_path, "error": str(e)}


def register_plugin(maki_instance=None) -> ImageClassifier:
    return ImageClassifier(maki_instance=maki_instance)
