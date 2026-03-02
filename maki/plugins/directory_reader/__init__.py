"""
Directory Reader Plugin for Maki Framework

This plugin provides functionality to read multiple text files from a directory
in the Maki framework.
"""

from .directory_reader import DirectoryReader, register_plugin

__all__ = ['DirectoryReader', 'register_plugin']
