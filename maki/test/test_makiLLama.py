#!/usr/bin/env python3
"""
Test script for the MakiLLama class.
This tests the functionality of the new LocalLLM wrapper that extends Maki.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_makillama_initialization():
    """Test that MakiLLama can be initialized with parameters"""
    try:
        from maki.makiLLama import MakiLLama

        # Test basic initialization
        llm = MakiLLama(model="gemma3")
        print("✓ MakiLLama initialization successful")

        # Test initialization with all parameters
        llm2 = MakiLLama(
            model="llama3",
            base_url="http://localhost:11434",
            system_prompt="You are a helpful assistant",
            timeout=60
        )
        print("✓ MakiLLama full initialization successful")

        # Test that attributes are set correctly
        assert llm.model == "gemma3"
        assert llm.base_url == "http://localhost:11434"
        assert llm.system_prompt is None
        assert llm.timeout == 120

        assert llm2.model == "llama3"
        assert llm2.base_url == "http://localhost:11434"
        assert llm2.system_prompt == "You are a helpful assistant"
        assert llm2.timeout == 60

        print("✓ MakiLLama attribute assignment successful")

        return True
    except Exception as e:
        print(f"✗ MakiLLama initialization failed: {e}")
        return False

def test_makillama_methods():
    """Test that MakiLLama has all the required methods"""
    try:
        from maki.makiLLama import MakiLLama

        llm = MakiLLama(model="gemma3")

        # Test that methods exist
        assert hasattr(llm, 'chat')
        assert hasattr(llm, 'stream')
        assert hasattr(llm, 'async_chat')
        assert hasattr(llm, 'session')
        assert hasattr(llm, 'list_models')
        assert hasattr(llm, 'pull')
        assert hasattr(llm, '_build_messages')
        assert hasattr(llm, '_verify_connection')
        print("✓ All MakiLLama methods present")

        return True
    except Exception as e:
        print(f"✗ MakiLLama methods test failed: {e}")
        return False

def test_makillama_chat_method():
    """Test the chat method functionality"""
    try:
        from maki.makiLLama import MakiLLama, Message, GenerationConfig

        # Mock requests to avoid actual HTTP calls
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:

            # Setup mock responses
            mock_get_response = MagicMock()
            mock_get_response.json.return_value = {"models": [{"name": "gemma3"}, {"name": "llama3"}]}
            mock_get_response.status_code = 200
            mock_get.return_value = mock_get_response

            mock_post_response = MagicMock()
            mock_post_response.json.return_value = {
                "message": {"content": "Test response"},
                "model": "gemma3",
                "prompt_eval_count": 10,
                "eval_count": 20
            }
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response

            llm = MakiLLama(model="gemma3")

            # Test chat method
            response = llm.chat("Test prompt")

            assert response.content == "Test response"
            assert response.model == "gemma3"
            assert response.prompt_tokens == 10
            assert response.completion_tokens == 20
            assert response.total_tokens == 30

            print("✓ MakiLLama chat method works correctly")

        return True
    except Exception as e:
        print(f"✗ MakiLLama chat test failed: {e}")
        return False

def test_makillama_stream_method():
    """Test the stream method functionality"""
    try:
        from maki.makiLLama import MakiLLama, Message

        # Mock requests to avoid actual HTTP calls
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:

            # Setup mock responses for get request (model verification)
            mock_get_response = MagicMock()
            mock_get_response.json.return_value = {"models": [{"name": "gemma3"}]}
            mock_get_response.status_code = 200
            mock_get.return_value = mock_get_response

            # Setup mock streaming response
            mock_post_response = MagicMock()
            mock_post_response.iter_lines.return_value = [
                b'{"message": {"content": "Hello"}}',
                b'{"message": {"content": " World"}}',
                b'{"done": true}'
            ]
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response

            llm = MakiLLama(model="gemma3")

            # Test stream method - should return a generator that yields chunks
            chunks = list(llm.stream("Test prompt"))
            assert len(chunks) == 2
            assert chunks[0] == "Hello"
            assert chunks[1] == " World"

            print("✓ MakiLLama stream method works correctly")

        return True
    except Exception as e:
        print(f"✗ MakiLLama stream test failed: {e}")
        return False

def test_makillama_session():
    """Test the session functionality"""
    try:
        from maki.makiLLama import MakiLLama, Message

        # Mock requests to avoid actual HTTP calls
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:

            # Setup mock responses
            mock_get_response = MagicMock()
            mock_get_response.json.return_value = {"models": [{"name": "gemma3"}]}
            mock_get_response.status_code = 200
            mock_get.return_value = mock_get_response

            mock_post_response = MagicMock()
            mock_post_response.json.return_value = {
                "message": {"content": "Test response"},
                "model": "gemma3",
                "prompt_eval_count": 10,
                "eval_count": 20
            }
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response

            llm = MakiLLama(model="gemma3")

            # Test session creation and usage
            session = llm.session(system="You are a helpful assistant.")
            assert session is not None

            # Test that session has the right methods
            assert hasattr(session, 'say')
            assert hasattr(session, 'reset')
            assert hasattr(session, 'print_history')
            assert hasattr(session, 'history')

            print("✓ MakiLLama session functionality works correctly")

        return True
    except Exception as e:
        print(f"✗ MakiLLama session test failed: {e}")
        return False

def test_makillama_generation_config():
    """Test GenerationConfig functionality"""
    try:
        from maki.makiLLama import GenerationConfig

        # Test default config
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.repeat_penalty == 1.1
        assert config.max_tokens == 2048
        assert config.seed == -1

        # Test custom config
        custom_config = GenerationConfig(
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            repeat_penalty=1.2,
            max_tokens=1024,
            seed=12345
        )

        assert custom_config.temperature == 0.5
        assert custom_config.top_p == 0.8
        assert custom_config.top_k == 30
        assert custom_config.repeat_penalty == 1.2
        assert custom_config.max_tokens == 1024
        assert custom_config.seed == 12345

        # Test to_ollama_options conversion
        options = custom_config.to_ollama_options()
        assert 'temperature' in options
        assert 'top_p' in options
        assert 'top_k' in options
        assert 'repeat_penalty' in options
        assert 'num_predict' in options

        print("✓ GenerationConfig works correctly")

        return True
    except Exception as e:
        print(f"✗ GenerationConfig test failed: {e}")
        return False

def test_makillama_message():
    """Test Message dataclass functionality"""
    try:
        from maki.makiLLama import Message

        # Test Message creation
        msg = Message(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"

        # Test to_dict method
        msg_dict = msg.to_dict()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello world"

        print("✓ Message dataclass works correctly")

        return True
    except Exception as e:
        print(f"✗ Message test failed: {e}")
        return False

def test_makillama_factories():
    """Test factory functions"""
    try:
        from maki.makiLLama import gemma3, qwen, llama, mistral

        # Test factory functions
        llm1 = gemma3()
        assert llm1.model == "gemma3"

        llm2 = qwen("qwen2.5:7b")
        assert llm2.model == "qwen2.5:7b"

        llm3 = llama("llama3.2")
        assert llm3.model == "llama3.2"

        llm4 = mistral()
        assert llm4.model == "mistral"

        print("✓ Factory functions work correctly")

        return True
    except Exception as e:
        print(f"✗ Factory functions test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MakiLLama class...")

    tests = [
        test_makillama_initialization,
        test_makillama_methods,
        test_makillama_chat_method,
        test_makillama_stream_method,
        test_makillama_session,
        test_makillama_generation_config,
        test_makillama_message,
        test_makillama_factories
    ]

    results = []
    for test in tests:
        results.append(test())

    if all(results):
        print("\n✓ All MakiLLama tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some MakiLLama tests failed!")
        sys.exit(1)