"""
File Mover Plugin for Maki Framework

Moves and renames files and directories identically on Linux, macOS and Windows, with a preview, a
monthly journal and a one-step undo.
"""

from .file_mover import FileMover, register_plugin

__all__ = ['FileMover', 'register_plugin']
