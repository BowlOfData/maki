# Testing MakiLLama Class

This project includes comprehensive tests for the `MakiLLama` class, which extends the base `Maki` class with additional features like streaming, async support, and session management.

## Test Files

1. **test_makiLLama.py** - Basic test script using print statements
2. **test_makiLLama_unittest.py** - Formal unit tests using Python's unittest framework

## Test Coverage

The tests cover the following functionality:

### Core Class Initialization
- Basic initialization with default parameters
- Full initialization with all parameters
- Attribute validation

### Methods Testing
- `chat()` method for single-turn generation
- `stream()` method for streaming generation
- `async_chat()` method for async operations
- `session()` method for multi-turn conversations
- `list_models()` and `pull()` methods for model management
- `_build_messages()` and `_verify_connection()` internal methods

### Data Structures
- `GenerationConfig` dataclass with proper defaults and conversion to Ollama options
- `Message` dataclass for conversation history
- Session management with proper history tracking

### Factory Functions
- `gemma3()`, `qwen()`, `llama()`, and `mistral()` convenience functions

## Running Tests

To run the tests, ensure you have the required dependencies installed:

```bash
pip install requests httpx rich
```

Then run either test file:

```bash
python3 test_makiLLama.py
# or
python3 test_makiLLama_unittest.py
```

Note: The tests will show warnings about models not being found locally, but this is expected and doesn't affect test execution.