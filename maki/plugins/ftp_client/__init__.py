"""
FTP Client Plugin for Maki Framework

This package contains the FTPClient plugin that provides functionality to connect
to FTP and SFTP servers for file operations.
"""

from .ftp_client import FTPClient, register_plugin

__all__ = ['FTPClient', 'register_plugin']