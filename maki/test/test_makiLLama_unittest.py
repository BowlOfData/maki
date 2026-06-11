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

from maki.makiLLama import MakiLLama, GenerationConfig, Message

class TestMakiLLama(unittest.TestCase):

    def test_initialization(self):
        """Test that MakiLLama can be initialized with parameters"""
        

        # Test basic initialization
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")

        self.assertEqual(llm.model, "gemma3")
        self.assertEqual(llm.base_url, "http://localhost:11434")
        self.assertIsNone(llm.system_prompt)
        self.assertEqual(llm.timeout, 120)

        # Test session creation and usage
        session = llm.session(system="You are a helpful assistant.")
        self.assertIsNotNone(session)


    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_chat_method(self, mock_get, mock_post):
        """Test the chat method functionality"""

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

    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_stream_method(self, mock_get, mock_post):
        """Test the stream method functionality"""

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

        # Test Message creation
        msg = Message(role="user", content="Hello world")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello world")

        # Test to_dict method
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["role"], "user")
        self.assertEqual(msg_dict["content"], "Hello world")

    @patch('requests.Session.post')
    def test_pull_progress_logging(self, mock_post):
        """pull() handles progress chunks with 'total' without crashing.

        Regression test: log.info(..., end='\\r') raised TypeError on the
        first progress chunk. Progress is now logged at ~10% intervals.
        """
        import json
        lines = [
            json.dumps({"status": "pulling manifest"}).encode(),
            json.dumps({"status": "downloading", "total": 100, "completed": 5}).encode(),
            json.dumps({"status": "downloading", "total": 100, "completed": 50}).encode(),
            json.dumps({"status": "downloading", "total": 100, "completed": 55}).encode(),
            json.dumps({"status": "downloading", "total": 100, "completed": 100}).encode(),
            json.dumps({"status": "success"}).encode(),
        ]
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")

        with self.assertLogs('maki.makiLLama', level='INFO') as cm:
            llm.pull()

        progress = [m for m in cm.output if '%]' in m]
        # 5% and 50% logged (>=10% apart), 55% suppressed, 100% logged.
        self.assertEqual(len(progress), 3)
        self.assertIn('[5%]', progress[0])
        self.assertIn('[50%]', progress[1])
        self.assertIn('[100%]', progress[2])

if __name__ == '__main__':
    unittest.main()