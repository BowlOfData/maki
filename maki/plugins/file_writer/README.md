# File Writer Plugin for Maki Framework

This plugin provides functionality to write text files in the Maki framework. It allows agents to write content to files with various options for handling existing files.

## Features

- Write text content to files with encoding support
- Append content to existing files
- Write lists of lines to files
- Create parent directories automatically
- Error handling for common file operations
- File information retrieval

## Usage

### Basic Usage

```python
from maki.plugins.file_writer import FileWriter

# Initialize the plugin
file_writer = FileWriter()

# Write content to a file
result = file_writer.write_file("path/to/your/file.txt", "Hello, World!")

# Check if successful
if result['success']:
    print(f"Successfully wrote {result['bytes_written']} bytes")
else:
    print("Error:", result['error'])
```

### Append to File

```python
# Append content to a file
result = file_writer.append_to_file("path/to/your/file.txt", "\nAdditional line")
```

### Write Lines

```python
# Write a list of lines to a file
lines = ["Line 1", "Line 2", "Line 3"]
result = file_writer.write_file_lines("path/to/your/file.txt", lines)
```

## Methods

### `write_file(file_path, content, encoding='utf-8', mode='w', create_dirs=True)`

Writes content to a text file.

**Parameters:**
- `file_path` (str): Path to the file to write
- `content` (str): Content to write to the file
- `encoding` (str): Text encoding to use (default: 'utf-8')
- `mode` (str): File mode ('w' for overwrite, 'a' for append, 'x' for exclusive creation)
- `create_dirs` (bool): Whether to create parent directories if they don't exist

**Returns:** Dictionary with write operation results

### `append_to_file(file_path, content, encoding='utf-8')`

Appends content to a text file.

**Parameters:**
- `file_path` (str): Path to the file to append to
- `content` (str): Content to append
- `encoding` (str): Text encoding to use (default: 'utf-8')

**Returns:** Dictionary with append operation results

### `write_file_lines(file_path, lines, encoding='utf-8', mode='w', create_dirs=True)`

Writes a list of lines to a text file.

**Parameters:**
- `file_path` (str): Path to the file to write
- `lines` (list): List of lines to write
- `encoding` (str): Text encoding to use (default: 'utf-8')
- `mode` (str): File mode ('w' for overwrite, 'a' for append, 'x' for exclusive creation)
- `create_dirs` (bool): Whether to create parent directories if they don't exist

**Returns:** Dictionary with write operation results

### `get_file_info(file_path)`

Gets information about a file.

**Parameters:**
- `file_path` (str): Path to the file

**Returns:** Dictionary with file information

## Error Handling

All methods return a dictionary with a `success` field. If `success` is `False`, an `error` field will contain the error message.

## Integration with Maki Agents

The plugin can be used by Maki agents to write file contents:

```python
from maki.maki import Maki
from maki.plugins.file_writer import FileWriter

# Initialize Maki and the plugin
maki = Maki("http://localhost", 11434, "llama3")
file_writer = FileWriter(maki)

# Use in an agent task
def save_analysis(agent, file_path, content):
    result = file_writer.write_file(file_path, content)
    if result['success']:
        return f"Successfully saved analysis to {file_path}"
    else:
        return f"Failed to save analysis: {result['error']}"
```