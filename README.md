# maki

A Python service for interacting with Ollama models. This library provides a simple interface to communicate with Ollama's API for generating text and retrieving model information.

## Features

- Send prompts to Ollama models
- Retrieve model version information
- Support for image-based prompts
- Simple and intuitive API

## Installation

```bash
pip install requests
```

## Usage

### Basic Setup

```python
from maki.maki import Maki

# Initialize the Maki object
maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)
```

### Generate Text

```python
# Simple text generation
prompt = "Explain quantum computing in simple terms"
response = maki.request(prompt)
print(response)
```

### With Images

```python
# Generate text with image input
response = maki.request_with_images("Describe this image", "path/to/image.jpg")
print(response)
```

### Get Model Version

```python
# Get the version of the connected model
version = maki.version()
print(version)
```

## API Reference

### Maki Class

- `__init__(url: str, port: str, model: str, temperature=0)`: Initialize the Maki object
- `request(prompt: str) -> str`: Send a prompt to the LLM and return the response
- `version() -> str`: Get the version of the connected LLM
- `request_with_images(prompt: str, img: str) -> str`: Send a prompt with image input

## Requirements

- Python 3.6+
- requests library

## License

MIT