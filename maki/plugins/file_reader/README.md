# File Reader Plugin for Maki Framework

This plugin provides functionality to read text files in the Maki framework. It allows agents to read file contents and return them for processing.

## Features

- Read text files with specified encoding
- Limit reading to a maximum number of lines
- Read files as lists of lines
- Get file information (size, existence, type)
- Error handling for common file operations

## Usage

### Basic Usage

```python
from maki.plugins.file_reader import FileReader

# Initialize the plugin
file_reader = FileReader()

# Read a file
result = file_reader.read_file("path/to/your/file.txt")

# Check if successful
if result['success']:
    print("File content:", result['content'])
else:
    print("Error:", result['error'])
```

### Reading as Lines

```python
# Read file as lines
result = file_reader.read_file_as_lines("path/to/your/file.txt")

if result['success']:
    for i, line in enumerate(result['lines']):
        print(f"Line {i+1}: {line.strip()}")
```

### Get File Information

```python
# Get file information
info = file_reader.get_file_info("path/to/your/file.txt")

if info['success']:
    print(f"File size: {info['size']} bytes")
    print(f"Is file: {info['is_file']}")
```

## Methods

### `read_file(file_path, encoding='utf-8', max_lines=None)`

Reads a text file and returns its contents.

**Parameters:**
- `file_path` (str): Path to the file to read
- `encoding` (str): Text encoding to use (default: 'utf-8')
- `max_lines` (int, optional): Maximum number of lines to read

**Returns:** Dictionary with file information and content

### `read_file_as_lines(file_path, encoding='utf-8')`

Reads a text file and returns its contents as a list of lines.

**Parameters:**
- `file_path` (str): Path to the file to read
- `encoding` (str): Text encoding to use (default: 'utf-8')

**Returns:** Dictionary with file information and lines

### `get_file_info(file_path)`

Gets information about a file.

**Parameters:**
- `file_path` (str): Path to the file

**Returns:** Dictionary with file information

## Error Handling

All methods return a dictionary with a `success` field. If `success` is `False`, an `error` field will contain the error message.

## Integration with Maki Agents

The plugin can be used by Maki agents to read and process file contents:

```python
from maki.maki import Maki
from maki.plugins.file_reader import FileReader

# Initialize Maki and the plugin
maki = Maki("http://localhost", 11434, "llama3")
file_reader = FileReader(maki)

# Use in an agent task
def process_file_task(agent, file_path):
    result = file_reader.read_file(file_path)
    if result['success']:
        # Process the file content with LLM
        prompt = f"Analyze this file content: {result['content']}"
        return agent.maki.request(prompt)
    else:
        return f"Failed to read file: {result['error']}"
```