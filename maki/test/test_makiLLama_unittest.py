#!/usr/bin/env python3
"""
Unit tests for the MakiLLama class using Python's unittest framework.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestMakiLLama(unittest.TestCase):

    def test_initialization(self):
        """Test that MakiLLama can be initialized with parameters"""
        from maki.makiLLama import MakiLLama

        # Test basic initialization
        llm = MakiLLama(model="gemma3")
        self.assertEqual(llm.model, "gemma3")
        self.assertEqual(llm.base_url, "http://localhost:11434")
        self.assertIsNone(llm.system_prompt)
        self.assertEqual(llm.timeout, 120)

        # Test full initialization
        llm2 = MakiLLama(
            model="llama3",
            base_url="http://localhost:11434",
            system_prompt="You are a helpful assistant",
            timeout=60
        )
        self.assertEqual(llm2.model, "llama3")
        self.assertEqual(llm2.base_url, "http://localhost:11434")
        self.assertEqual(llm2.system_prompt, "You are a helpful assistant")
        self.assertEqual(llm2.timeout, 60)

    def test_methods_exist(self):
        """Test that MakiLLama has all the required methods"""
        from maki.makiLLama import MakiLLama

        llm = MakiLLama(model="gemma3")

        # Test that methods exist
        self.assertTrue(hasattr(llm, 'chat'))
        self.assertTrue(hasattr(llm, 'stream'))
        self.assertTrue(hasattr(llm, 'async_chat'))
        self.assertTrue(hasattr(llm, 'session'))
        self.assertTrue(hasattr(llm, 'list_models'))
        self.assertTrue(hasattr(llm, 'pull'))
        self.assertTrue(hasattr(llm, '_build_messages'))
        self.assertTrue(hasattr(llm, '_verify_connection'))

    @patch('requests.post')
    @patch('requests.get')
    def test_chat_method(self, mock_get, mock_post):
        """Test the chat method functionality"""
        from maki.makiLLama import MakiLLama

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

        # Mock the _verify_connection method to avoid localhost connection issues
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")

        # Test chat method
        response = llm.chat("Test prompt")

        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "gemma3")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 20)
        self.assertEqual(response.total_tokens, 30)

    @patch('requests.post')
    @patch('requests.get')
    def test_stream_method(self, mock_get, mock_post):
        """Test the stream method functionality"""
        from maki.makiLLama import MakiLLama

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

        # Mock the _verify_connection method to avoid localhost connection issues
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")

        # Test stream method - should return a generator that yields chunks
        chunks = list(llm.stream("Test prompt"))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Hello")
        self.assertEqual(chunks[1], " World")

    @patch('requests.post')
    @patch('requests.get')
    def test_session_functionality(self, mock_get, mock_post):
        """Test the session functionality"""
        from maki.makiLLama import MakiLLama

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

        # Mock the _verify_connection method to avoid localhost connection issues
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")

        # Test session creation and usage
        session = llm.session(system="You are a helpful assistant.")
        self.assertIsNotNone(session)

        # Test that session has the right methods
        self.assertTrue(hasattr(session, 'say'))
        self.assertTrue(hasattr(session, 'reset'))
        self.assertTrue(hasattr(session, 'print_history'))
        self.assertTrue(hasattr(session, 'history'))

    def test_generation_config(self):
        """Test GenerationConfig functionality"""
        from maki.makiLLama import GenerationConfig

        # Test default config
        config = GenerationConfig()
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 40)
        self.assertEqual(config.repeat_penalty, 1.1)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.seed, -1)

        # Test custom config
        custom_config = GenerationConfig(
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            repeat_penalty=1.2,
            max_tokens=1024,
            seed=12345
        )

        self.assertEqual(custom_config.temperature, 0.5)
        self.assertEqual(custom_config.top_p, 0.8)
        self.assertEqual(custom_config.top_k, 30)
        self.assertEqual(custom_config.repeat_penalty, 1.2)
        self.assertEqual(custom_config.max_tokens, 1024)
        self.assertEqual(custom_config.seed, 12345)

        # Test to_ollama_options conversion
        options = custom_config.to_ollama_options()
        self.assertIn('temperature', options)
        self.assertIn('top_p', options)
        self.assertIn('top_k', options)
        self.assertIn('repeat_penalty', options)
        self.assertIn('num_predict', options)

    def test_message_dataclass(self):
        """Test Message dataclass functionality"""
        from maki.makiLLama import Message

        # Test Message creation
        msg = Message(role="user", content="Hello world")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello world")

        # Test to_dict method
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["role"], "user")
        self.assertEqual(msg_dict["content"], "Hello world")

    def test_factory_functions(self):
        """Test factory functions"""
        from maki.makiLLama import gemma3, qwen, llama, mistral

        # Test factory functions
        llm1 = gemma3()
        self.assertEqual(llm1.model, "gemma3")

        llm2 = qwen("qwen2.5:7b")
        self.assertEqual(llm2.model, "qwen2.5:7b")

        llm3 = llama("llama3.2")
        self.assertEqual(llm3.model, "llama3.2")

        llm4 = mistral()
        self.assertEqual(llm4.model, "mistral")

if __name__ == '__main__':
    unittest.main()