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

    def __init__(self, maki_instance=None, base_dir: str = None):
        """
        Initialize the FileWriter plugin.

        Args:
            maki_instance: Optional Maki instance for logging and LLM interactions
            base_dir: Root directory that all file paths must resolve within.
                      Defaults to the current working directory. Symlinks in
                      this path are resolved so that link-based escapes are
                      caught consistently.
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.base_dir = os.path.realpath(base_dir if base_dir is not None else os.getcwd())
        self.logger.info(f"FileWriter plugin initialized (base_dir='{self.base_dir}')")

    def _safe_path(self, path: str) -> str:
        """
        Resolve *path* within self.base_dir and verify it does not escape.

        Symlinks are fully resolved so that a link pointing outside base_dir
        is caught the same way as a plain ``../../`` traversal.

        Args:
            path: A relative or absolute file/directory path supplied by the caller.

        Returns:
            The canonicalised absolute path, guaranteed to be inside base_dir.

        Raises:
            ValueError: If the resolved path is outside base_dir.
        """
        resolved = os.path.realpath(os.path.join(self.base_dir, path))
        if resolved != self.base_dir and not resolved.startswith(self.base_dir + os.sep):
            raise ValueError(
                f"Path '{path}' resolves outside the allowed directory '{self.base_dir}'"
            )
        return resolved

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
            safe = self._safe_path(file_path)
        except ValueError as exc:
            result['error'] = str(exc)
            self.logger.warning(str(exc))
            return result

        try:
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(safe)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

            # Write the file
            with open(safe, mode, encoding=encoding) as file:
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
            safe = self._safe_path(file_path)
        except ValueError as exc:
            result['error'] = str(exc)
            self.logger.warning(str(exc))
            return result

        try:
            # Append to the file
            with open(safe, 'a', encoding=encoding) as file:
                bytes_written = file.write(content)

            result['success'] = True
            result['bytes_written'] = bytes_written

            self.logger.info(f"Successfully appended to file: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error appending to file {file_path}: {str(e)}", exc_info=True)

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
            safe = self._safe_path(file_path)
        except ValueError as exc:
            result['error'] = str(exc)
            self.logger.warning(str(exc))
            return result

        try:
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(safe)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

            # Write the lines to file
            with open(safe, mode, encoding=encoding) as file:
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
            safe = self._safe_path(file_path)
        except ValueError as exc:
            result['error'] = str(exc)
            self.logger.warning(str(exc))
            return result

        try:
            result['exists'] = os.path.exists(safe)
            result['is_file'] = os.path.isfile(safe)
            result['is_directory'] = os.path.isdir(safe)

            if result['exists'] and result['is_file']:
                result['size'] = os.path.getsize(safe)
                result['success'] = True

            self.logger.info(f"Retrieved file info for: {file_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")

        return result


# Plugin registration function
def register_plugin(maki_instance=None, base_dir: str = None):
    """
    Register the FileWriter plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        FileWriter: An instance of the FileWriter plugin
    """
    return FileWriter(maki_instance, base_dir=base_dir)