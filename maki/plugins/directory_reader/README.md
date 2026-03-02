# Directory Reader Plugin for Maki Framework

This plugin provides functionality to read multiple text files from a directory in the Maki framework. It is built on top of the existing `file_reader` plugin and reuses its single-file reading logic.

## Features

- Read all text files from a directory
- Optionally read directories recursively
- Filter files by extension
- Limit the number of files read
- Limit the number of lines read per file
- Aggregate directory contents into a single text block
- Skip hidden files by default

## Usage

### Basic Usage

```python
from maki.plugins.directory_reader import DirectoryReader

directory_reader = DirectoryReader()
result = directory_reader.read_directory("path/to/your/folder")

if result['success']:
    print("Files read:", result['read_files'])
else:
    print("Error:", result['error'])
```

### Recursive Reading with Filters

```python
result = directory_reader.read_directory(
    "path/to/your/project",
    recursive=True,
    extensions=[".py", ".md"],
    max_files=20,
    max_lines_per_file=200,
)
```

### Aggregate Content for LLM Processing

```python
result = directory_reader.read_directory_as_text(
    "path/to/your/project",
    recursive=True,
    extensions=[".py"],
)

if result['success']:
    print(result['content'])
```

## Methods

### `read_directory(dir_path, encoding='utf-8', recursive=False, extensions=None, max_files=None, max_lines_per_file=None, include_hidden=False)`

Reads matching files from a directory and returns a structured result.

### `read_directory_as_text(dir_path, encoding='utf-8', recursive=False, extensions=None, max_files=None, max_lines_per_file=None, include_hidden=False)`

Reads matching files from a directory and returns both structured results and a single aggregated text block.
