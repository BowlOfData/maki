"""
File Writer Plugin for Maki Framework

This plugin provides functionality to write text files in the Maki framework.
It allows agents to write content to files with various options for handling existing files.
"""

import os
import logging
from typing import Dict, Any, Optional


class FileWriter:
    """
    A plugin class for writing text files in the Maki framework.

    This class provides methods to write text content to files with various
    options for handling existing files and encoding.
    """

    def __init__(self, maki_instance=None):
        """
        Initialize the FileWriter plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and potential LLM interactions
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("FileWriter plugin initialized")

    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8',
                   mode: str = 'w', create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write content to a text file.

        Args:
            file_path (str): The path to the file to write
            content (str): The content to write to the file
            encoding (str): The text encoding to use (default: 'utf-8')
            mode (str): The file mode ('w' for overwrite, 'a' for append, 'x' for exclusive creation)
            create_dirs (bool): Whether to create parent directories if they don't exist

        Returns:
            Dict[str, Any]: A dictionary containing write operation results
                - 'success': Boolean indicating if write was successful
                - 'file_path': The path of the file written
                - 'bytes_written': Number of bytes written
                - 'encoding': The encoding used
                - 'mode': The mode used
                - 'error': Error message if write failed

        Raises:
            ValueError: If file_path or content is not a valid string
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        if not isinstance(content, str):
            raise ValueError("content must be a string")

        result = {
            'success': False,
            'file_path': file_path,
            'bytes_written': 0,
            'encoding': encoding,
            'mode': mode,
            'error': None
        }

        try:
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

            # Write the file
            with open(file_path, mode, encoding=encoding) as file:
                bytes_written = file.write(content)

            result['success'] = True
            result['bytes_written'] = bytes_written

            self.logger.info(f"Successfully wrote to file: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error writing to file {file_path}: {str(e)}")

        return result

    def append_to_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Append content to a text file.

        Args:
            file_path (str): The path to the file to append to
            content (str): The content to append to the file
            encoding (str): The text encoding to use (default: 'utf-8')

        Returns:
            Dict[str, Any]: A dictionary containing append operation results
                - 'success': Boolean indicating if append was successful
                - 'file_path': The path of the file appended to
                - 'bytes_written': Number of bytes written
                - 'encoding': The encoding used
                - 'error': Error message if append failed

        Raises:
            ValueError: If file_path or content is not a valid string
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        if not isinstance(content, str):
            raise ValueError("content must be a string")

        result = {
            'success': False,
            'file_path': file_path,
            'bytes_written': 0,
            'encoding': encoding,
            'error': None
        }

        try:
            # Append to the file
            with open(file_path, 'a', encoding=encoding) as file:
                bytes_written = file.write(content)

            result['success'] = True
            result['bytes_written'] = bytes_written

            self.logger.info(f"Successfully appended to file: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error appending to file {file_path}: {str(e)}")

        return result

    def write_file_lines(self, file_path: str, lines: list, encoding: str = 'utf-8',
                         mode: str = 'w', create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write a list of lines to a text file.

        Args:
            file_path (str): The path to the file to write
            lines (list): List of lines to write to the file
            encoding (str): The text encoding to use (default: 'utf-8')
            mode (str): The file mode ('w' for overwrite, 'a' for append, 'x' for exclusive creation)
            create_dirs (bool): Whether to create parent directories if they don't exist

        Returns:
            Dict[str, Any]: A dictionary containing write operation results
                - 'success': Boolean indicating if write was successful
                - 'file_path': The path of the file written
                - 'lines_written': Number of lines written
                - 'encoding': The encoding used
                - 'mode': The mode used
                - 'error': Error message if write failed

        Raises:
            ValueError: If file_path is not a valid string or lines is not a list
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        if not isinstance(lines, list):
            raise ValueError("lines must be a list")

        result = {
            'success': False,
            'file_path': file_path,
            'lines_written': 0,
            'encoding': encoding,
            'mode': mode,
            'error': None
        }

        try:
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

            # Write the lines to file
            with open(file_path, mode, encoding=encoding) as file:
                lines_written = 0
                for line in lines:
                    file.write(str(line) + '\n')
                    lines_written += 1

            result['success'] = True
            result['lines_written'] = lines_written

            self.logger.info(f"Successfully wrote {lines_written} lines to file: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error writing lines to file {file_path}: {str(e)}")

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
    Register the FileWriter plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        FileWriter: An instance of the FileWriter plugin
    """
    return FileWriter(maki_instance)