# Web to Markdown Plugin

This plugin provides functionality to fetch web pages and convert them to markdown format. It uses the existing file_writer plugin to save the output to files.

## Features

- Fetch web pages from URLs
- Convert HTML content to markdown format
- Save markdown output to files using the file_writer plugin
- Automatic filename generation based on URL
- Error handling for network and file operations

## Usage

### Basic Usage

```python
from maki.maki import Maki
from maki.plugins.web_to_md.web_to_md import WebToMd

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the web_to_md plugin
web_to_md = WebToMd(maki)

# Fetch a web page and convert to markdown
result = web_to_md.fetch_and_convert_to_md("https://example.com")
if result['success']:
    print(f"Successfully saved to {result['output_file']}")
    print(f"Content: {result['content'][:200]}...")
else:
    print(f"Error: {result['error']}")
```

### With Custom Output File

```python
# Fetch and save to a specific file
result = web_to_md.fetch_and_convert_to_md("https://example.com", "output.md")
```

## Methods

### `fetch_and_convert_to_md(url, output_file=None)`

Fetches a web page and converts it to markdown format.

**Parameters:**
- `url` (str): The URL of the web page to fetch
- `output_file` (str, optional): The path to save the markdown output. If None, generates a filename based on the URL.

**Returns:**
A dictionary containing:
- `success` (bool): Whether the operation was successful
- `url` (str): The URL that was fetched
- `output_file` (str): The path of the output file
- `content` (str): The markdown content (if successful)
- `error` (str): Error message if operation failed
- `status_code` (int): HTTP status code (if applicable)