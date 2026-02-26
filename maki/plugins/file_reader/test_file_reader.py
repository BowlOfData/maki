"""
Test file for the FileReader plugin
"""

import os
import tempfile
from maki.plugins.file_reader.file_reader import FileReader


def test_file_reader():
    """Test the FileReader plugin functionality"""

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        temp_file_path = f.name

    try:
        # Initialize the plugin
        file_reader = FileReader()

        # Test reading the file
        result = file_reader.read_file(temp_file_path)
        print("Reading file result:", result)

        assert result['success'] == True
        assert result['file_path'] == temp_file_path
        assert 'Line 1' in result['content']
        assert result['line_count'] == 5

        # Test reading as lines
        lines_result = file_reader.read_file_as_lines(temp_file_path)
        print("Reading as lines result:", lines_result)

        assert lines_result['success'] == True
        assert len(lines_result['lines']) == 5

        # Test getting file info
        info_result = file_reader.get_file_info(temp_file_path)
        print("File info result:", info_result)

        assert info_result['success'] == True
        assert info_result['is_file'] == True
        assert info_result['exists'] == True
        assert info_result['size'] > 0

        # Test with non-existent file
        non_existent_result = file_reader.read_file("/non/existent/file.txt")
        print("Non-existent file result:", non_existent_result)

        assert non_existent_result['success'] == False
        assert 'not found' in non_existent_result['error'].lower()

        print("All tests passed!")

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    test_file_reader()