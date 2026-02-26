"""
File Reader Plugin for Maki Framework

This plugin provides functionality to read text files in the Maki framework.
It allows agents to read file contents and return them for processing.
"""

from .file_reader import FileReader, register_plugin

__all__ = ['FileReader', 'register_plugin']
