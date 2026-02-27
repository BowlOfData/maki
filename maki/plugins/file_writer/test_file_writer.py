"""
Test file for the FileWriter plugin
"""

import os
import tempfile
from maki.plugins.file_writer.file_writer import FileWriter


def test_file_writer():
    """Test the FileWriter plugin functionality"""

    # Initialize the plugin
    file_writer = FileWriter()

    # Test writing to a file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file_path = f.name

    try:
        # Test basic write
        result = file_writer.write_file(temp_file_path, "Hello, World!\nLine 2")
        print("Write file result:", result)

        assert result['success'] == True
        assert result['file_path'] == temp_file_path
        assert result['bytes_written'] > 0

        # Test reading back to verify
        with open(temp_file_path, 'r') as f:
            content = f.read()
        assert "Hello, World!" in content

        # Test append
        append_result = file_writer.append_to_file(temp_file_path, "\nAppended line")
        print("Append result:", append_result)

        assert append_result['success'] == True
        assert append_result['bytes_written'] > 0

        # Test reading back to verify append
        with open(temp_file_path, 'r') as f:
            content = f.read()
        assert "Appended line" in content

        # Test writing lines
        lines_result = file_writer.write_file_lines(temp_file_path, ["Line 1", "Line 2", "Line 3"])
        print("Write lines result:", lines_result)

        assert lines_result['success'] == True
        assert lines_result['lines_written'] == 3

        # Test getting file info
        info_result = file_writer.get_file_info(temp_file_path)
        print("File info result:", info_result)

        assert info_result['success'] == True
        assert info_result['is_file'] == True
        assert info_result['exists'] == True
        assert info_result['size'] > 0

        # Test with non-existent directory (should create it)
        temp_dir = tempfile.mkdtemp()
        new_file_path = os.path.join(temp_dir, "subdir", "new_file.txt")

        dir_result = file_writer.write_file(new_file_path, "Content in subdirectory")
        print("Directory creation result:", dir_result)

        assert dir_result['success'] == True
        assert os.path.exists(new_file_path)

        # Clean up
        os.unlink(new_file_path)
        os.rmdir(os.path.join(temp_dir, "subdir"))
        os.rmdir(temp_dir)

        print("All tests passed!")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    test_file_writer()