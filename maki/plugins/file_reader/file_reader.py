"""
File Reader Plugin for Maki Framework

This module provides functionality to read text files in the Maki framework.
It allows agents to read file contents and return them for processing.
"""

import os
import logging
from typing import Dict, Any, Optional


class FileReader:
    """
    A plugin class for reading text files in the Maki framework.

    This class provides methods to read text files and return their contents
    for processing by Maki agents.
    """

    def __init__(self, maki_instance=None):
        """
        Initialize the FileReader plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and potential LLM interactions
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("FileReader plugin initialized")

    def read_file(self, file_path: str, encoding: str = 'utf-8', max_lines: Optional[int] = None) -> Dict[str, Any]:
        """
        Read a text file and return its contents.

        Args:
            file_path (str): The path to the file to read
            encoding (str): The text encoding to use (default: 'utf-8')
            max_lines (int, optional): Maximum number of lines to read. If None, reads entire file.

        Returns:
            Dict[str, Any]: A dictionary containing file information and content
                - 'success': Boolean indicating if read was successful
                - 'file_path': The path of the file read
                - 'content': The file content
                - 'encoding': The encoding used
                - 'line_count': Number of lines in the file
                - 'error': Error message if read failed

        Raises:
            ValueError: If file_path is not a valid string
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        result = {
            'success': False,
            'file_path': file_path,
            'content': '',
            'encoding': encoding,
            'line_count': 0,
            'error': None
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result['error'] = f"File not found: {file_path}"
                self.logger.error(f"File not found: {file_path}")
                return result

            # Check if it's actually a file (not a directory)
            if not os.path.isfile(file_path):
                result['error'] = f"Path is not a file: {file_path}"
                self.logger.error(f"Path is not a file: {file_path}")
                return result

            # Read the file
            with open(file_path, 'r', encoding=encoding) as file:
                if max_lines is not None:
                    lines = []
                    for i, line in enumerate(file):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                else:
                    content = file.read()

            result['success'] = True
            result['content'] = content
            result['line_count'] = len(content.splitlines())

            self.logger.info(f"Successfully read file: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading file {file_path}: {str(e)}")

        return result

    def read_file_as_lines(self, file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read a text file and return its contents as a list of lines.

        Args:
            file_path (str): The path to the file to read
            encoding (str): The text encoding to use (default: 'utf-8')

        Returns:
            Dict[str, Any]: A dictionary containing file information and lines
                - 'success': Boolean indicating if read was successful
                - 'file_path': The path of the file read
                - 'lines': List of lines in the file
                - 'line_count': Number of lines in the file
                - 'error': Error message if read failed

        Raises:
            ValueError: If file_path is not a valid string
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        result = {
            'success': False,
            'file_path': file_path,
            'lines': [],
            'line_count': 0,
            'error': None
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result['error'] = f"File not found: {file_path}"
                self.logger.error(f"File not found: {file_path}")
                return result

            # Check if it's actually a file (not a directory)
            if not os.path.isfile(file_path):
                result['error'] = f"Path is not a file: {file_path}"
                self.logger.error(f"Path is not a file: {file_path}")
                return result

            # Read the file as lines
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()

            result['success'] = True
            result['lines'] = lines
            result['line_count'] = len(lines)

            self.logger.info(f"Successfully read file as lines: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading file {file_path} as lines: {str(e)}")

        return result

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.

        Args:
            file_path (str): The path to the file

        Returns:
            Dict[str, Any]: A dictionary containing file information
                - 'success': Boolean indicating if operation was successful
                - 'file_path': The path of the file
                - 'size': File size in bytes
                - 'is_file': Boolean indicating if path is a file
                - 'is_directory': Boolean indicating if path is a directory
                - 'exists': Boolean indicating if file exists
                - 'error': Error message if operation failed
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        result = {
            'success': False,
            'file_path': file_path,
            'size': 0,
            'is_file': False,
            'is_directory': False,
            'exists': False,
            'error': None
        }

        try:
            result['exists'] = os.path.exists(file_path)
            result['is_file'] = os.path.isfile(file_path)
            result['is_directory'] = os.path.isdir(file_path)

            if result['exists'] and result['is_file']:
                result['size'] = os.path.getsize(file_path)
                result['success'] = True

            self.logger.info(f"Retrieved file info for: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")

        return result


# Plugin registration function
def register_plugin(maki_instance=None):
    """
    Register the FileReader plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        FileReader: An instance of the FileReader plugin
    """
    return FileReader(maki_instance)