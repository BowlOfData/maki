"""
Test file for the DirectoryReader plugin
"""

import os
import tempfile

from maki.plugins.directory_reader.directory_reader import DirectoryReader


def test_directory_reader():
    """Test the DirectoryReader plugin functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        root_text_path = os.path.join(temp_dir, "root.txt")
        root_markdown_path = os.path.join(temp_dir, "notes.md")
        hidden_path = os.path.join(temp_dir, ".hidden.txt")
        nested_dir = os.path.join(temp_dir, "nested")
        nested_text_path = os.path.join(nested_dir, "nested.txt")

        os.mkdir(nested_dir)

        with open(root_text_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("Line 1\nLine 2\nLine 3")

        with open(root_markdown_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("# Title\n\nMarkdown content")

        with open(hidden_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("Hidden content")

        with open(nested_text_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("Nested line 1\nNested line 2")

        directory_reader = DirectoryReader()

        result = directory_reader.read_directory(temp_dir)
        print("Reading directory result:", result)

        assert result['success'] == True
        assert result['read_files'] == 2
        assert result['failed_files'] == 0
        assert result['total_files'] == 2

        recursive_result = directory_reader.read_directory(
            temp_dir,
            recursive=True,
            extensions=['.txt']
        )
        print("Recursive directory result:", recursive_result)

        assert recursive_result['success'] == True
        assert recursive_result['read_files'] == 2
        assert recursive_result['total_files'] == 2
        assert all(
            file_result['file_path'].endswith('.txt')
            for file_result in recursive_result['files']
        )

        aggregated_result = directory_reader.read_directory_as_text(
            temp_dir,
            recursive=True,
            extensions=['.txt'],
            max_lines_per_file=1
        )
        print("Aggregated directory result:", aggregated_result)

        assert aggregated_result['success'] == True
        assert aggregated_result['content_file_count'] == 2
        assert "=== " in aggregated_result['content']
        assert "Line 1" in aggregated_result['content']
        assert "Nested line 1" in aggregated_result['content']

        non_existent_result = directory_reader.read_directory("/non/existent/folder")
        print("Non-existent directory result:", non_existent_result)

        assert non_existent_result['success'] == False
        assert 'not found' in non_existent_result['error'].lower()

        print("All tests passed!")


if __name__ == "__main__":
    test_directory_reader()
