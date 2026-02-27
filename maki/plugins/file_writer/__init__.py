"""
File Writer Plugin for Maki Framework

This plugin provides functionality to write text files in the Maki framework.
It allows agents to write content to files with various options for handling existing files.
"""

from .file_writer import FileWriter, register_plugin

__all__ = ['FileWriter', 'register_plugin']
