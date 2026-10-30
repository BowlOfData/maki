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

    # Methods callable by the LLM via TOOL: directives (read-only).
    ALLOWED_METHODS = FileReader.ALLOWED_METHODS + [
        "read_directory",
        "read_directory_as_text",
    ]

    def __init__(self, maki_instance=None, base_dir: str = None):
        """
        Initialize the DirectoryReader plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and
                potential LLM interactions
            base_dir: Root directory that all paths must resolve within.
                      Defaults to the current working directory.
        """
        super().__init__(maki_instance, base_dir=base_dir)
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
            safe_dir = self._safe_path(dir_path)
        except ValueError as exc:
            result['error'] = str(exc)
            self.logger.warning(str(exc))
            return result

        try:
            problem = self._directory_problem(safe_dir, dir_path)
            if problem:
                result['error'] = problem
                self.logger.error(problem)
                return result

            self._read_matching_files(result, safe_dir, recursive, encoding, normalized_extensions,
                                      max_files, max_lines_per_file, include_hidden)

            result['success'] = True
            self.logger.info(f"Successfully read directory: {dir_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading directory {dir_path}: {str(e)}", exc_info=True)

        return result

    def _read_matching_files(self, result: Dict[str, Any], safe_dir: str, recursive: bool, encoding: str,
                             extensions, max_files: Optional[int], max_lines_per_file: Optional[int],
                             include_hidden: bool) -> None:
        """Read every matching file under ``safe_dir`` into ``result`` (files, counters)."""
        for entry_path in self._collect_entries(safe_dir, recursive):
            if self._skip_entry(entry_path, include_hidden, extensions):
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

    @staticmethod
    def _directory_problem(safe_dir: str, dir_path: str) -> Optional[str]:
        if not os.path.exists(safe_dir):
            return f"Directory not found: {dir_path}"
        if not os.path.isdir(safe_dir):
            return f"Path is not a directory: {dir_path}"
        return None

    @staticmethod
    def _skip_entry(entry_path: str, include_hidden: bool, extensions) -> bool:
        """True for hidden entries (unless included), non-files and files outside ``extensions``."""
        if not include_hidden and os.path.basename(entry_path).startswith('.'):
            return True
        if not os.path.isfile(entry_path):
            return True
        return bool(extensions) and os.path.splitext(entry_path)[1].lower() not in extensions

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


def register_plugin(maki_instance=None, base_dir: str = None):
    """
    Register the DirectoryReader plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin
        base_dir: Root directory that all paths must resolve within.

    Returns:
        DirectoryReader: An instance of the DirectoryReader plugin
    """
    return DirectoryReader(maki_instance, base_dir=base_dir)
