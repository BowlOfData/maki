"""
Example usage of the FileWriter plugin with Maki agents
"""

from maki.maki import Maki
from maki.plugins.file_writer.file_writer import FileWriter

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the file writer plugin
file_writer = FileWriter(maki)

# Example: Write content to a file
def save_content(file_path, content):
    # Write content to file
    result = file_writer.write_file(file_path, content)

    if result['success']:
        return f"Successfully wrote {result['bytes_written']} bytes to {file_path}"
    else:
        return f"Failed to write file: {result['error']}"

# Example: Append to a file
def append_content(file_path, content):
    # Append content to file
    result = file_writer.append_to_file(file_path, content)

    if result['success']:
        return f"Successfully appended {result['bytes_written']} bytes to {file_path}"
    else:
        return f"Failed to append to file: {result['error']}"

# Example: Write lines to a file
def save_lines(file_path, lines):
    # Write lines to file
    result = file_writer.write_file_lines(file_path, lines)

    if result['success']:
        return f"Successfully wrote {result['lines_written']} lines to {file_path}"
    else:
        return f"Failed to write lines: {result['error']}"

if __name__ == "__main__":
    print("FileWriter plugin example usage")
    print("================================")
    print("Plugin is ready to be used with Maki agents")