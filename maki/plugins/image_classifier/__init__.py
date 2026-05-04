"""
Image Classifier Plugin for Maki Framework

Classifies images using a vision-capable Ollama model. Supports both
synchronous and asynchronous invocation. The prompt and optional system
prompt are supplied by the caller, keeping the plugin domain-agnostic.
"""

from .image_classifier import ImageClassifier, register_plugin

__all__ = ["ImageClassifier", "register_plugin"]
