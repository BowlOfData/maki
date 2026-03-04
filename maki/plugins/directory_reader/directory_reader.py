"""
Directory Reader Plugin for Maki Framework

This module provides functionality to read directories of text files in the
Maki framework. It builds on the existing FileReader plugin and reuses its
single-file reading logic.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from maki.plugins.file_reader.file_reader import FileReader


class DirectoryReader(FileReader):
    """
    A plugin class for reading text files from directories in the Maki framework.

    This class extends FileReader and adds methods to collect and read multiple
    files from a directory, optionally recursively.
    """

    def __init__(self, maki_instance=None):
        """
        Initialize the DirectoryReader plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and
                potential LLM interactions
        """
        super().__init__(maki_instance)
        self.logger = logging.getLogger(__name__)
        self.logger.info("DirectoryReader plugin initialized")

    def _normalize_extensions(self, extensions: Optional[List[str]]) -> List[str]:
        """Normalize extension filters to lowercase values with a leading dot."""
        if not extensions:
            return []

        normalized = []
        for extension in extensions:
            if not isinstance(extension, str) or not extension.strip():
                continue
            cleaned = extension.strip().lower()
            if not cleaned.startswith('.'):
                cleaned = f".{cleaned}"
            normalized.append(cleaned)

        return normalized

    def _collect_entries(self, dir_path: str, recursive: bool) -> List[str]:
        """Collect candidate paths from a directory."""
        entries: List[str] = []

        if recursive:
            for root, _, filenames in os.walk(dir_path):
                for filename in sorted(filenames):
                    entries.append(os.path.join(root, filename))
        else:
            for entry in sorted(os.listdir(dir_path)):
                entries.append(os.path.join(dir_path, entry))

        return entries

    def read_directory(
        self,
        dir_path: str,
        encoding: str = 'utf-8',
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_lines_per_file: Optional[int] = None,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        Read all matching text files from a directory.

        Args:
            dir_path (str): The path to the directory to read
            encoding (str): The text encoding to use (default: 'utf-8')
            recursive (bool): Whether to read files recursively
            extensions (list[str], optional): Limit reading to these file
                extensions (for example ['.py', '.md'])
            max_files (int, optional): Maximum number of files to read
            max_lines_per_file (int, optional): Maximum number of lines to read
                from each file
            include_hidden (bool): Whether to include hidden files

        Returns:
            Dict[str, Any]: A dictionary containing directory read results
        """
        if not isinstance(dir_path, str) or not dir_path.strip():
            raise ValueError("dir_path must be a non-empty string")

        if max_files is not None and max_files < 1:
            raise ValueError("max_files must be greater than 0")

        normalized_extensions = self._normalize_extensions(extensions)

        result = {
            'success': False,
            'directory_path': dir_path,
            'files': [],
            'total_files': 0,
            'read_files': 0,
            'failed_files': 0,
            'skipped_entries': 0,
            'recursive': recursive,
            'extensions': normalized_extensions,
            'error': None
        }

        try:
            if not os.path.exists(dir_path):
                result['error'] = f"Directory not found: {dir_path}"
                self.logger.error(f"Directory not found: {dir_path}")
                return result

            if not os.path.isdir(dir_path):
                result['error'] = f"Path is not a directory: {dir_path}"
                self.logger.error(f"Path is not a directory: {dir_path}")
                return result

            for entry_path in self._collect_entries(dir_path, recursive):
                entry_name = os.path.basename(entry_path)

                if not include_hidden and entry_name.startswith('.'):
                    result['skipped_entries'] += 1
                    continue

                if not os.path.isfile(entry_path):
                    result['skipped_entries'] += 1
                    continue

                if normalized_extensions:
                    entry_extension = os.path.splitext(entry_path)[1].lower()
                    if entry_extension not in normalized_extensions:
                        result['skipped_entries'] += 1
                        continue

                if max_files is not None and result['total_files'] >= max_files:
                    break

                file_result = self.read_file(
                    entry_path,
                    encoding=encoding,
                    max_lines=max_lines_per_file
                )
                result['files'].append(file_result)
                result['total_files'] += 1

                if file_result['success']:
                    result['read_files'] += 1
                else:
                    result['failed_files'] += 1

            result['success'] = True
            self.logger.info(f"Successfully read directory: {dir_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading directory {dir_path}: {str(e)}", exc_info=True)

        return result

    def read_directory_as_text(
        self,
        dir_path: str,
        encoding: str = 'utf-8',
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_lines_per_file: Optional[int] = None,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        Read a directory and aggregate file contents into a single text block.

        Returns:
            Dict[str, Any]: The directory result plus an aggregated content field
        """
        result = self.read_directory(
            dir_path=dir_path,
            encoding=encoding,
            recursive=recursive,
            extensions=extensions,
            max_files=max_files,
            max_lines_per_file=max_lines_per_file,
            include_hidden=include_hidden
        )

        result['content'] = ''
        result['content_file_count'] = 0

        if not result['success']:
            return result

        sections = []
        for file_result in result['files']:
            if not file_result['success']:
                continue

            sections.append(
                f"=== {file_result['file_path']} ===\n{file_result['content']}"
            )

        result['content'] = "\n\n".join(sections)
        result['content_file_count'] = len(sections)
        return result


def register_plugin(maki_instance=None):
    """
    Register the DirectoryReader plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        DirectoryReader: An instance of the DirectoryReader plugin
    """
    return DirectoryReader(maki_instance)
