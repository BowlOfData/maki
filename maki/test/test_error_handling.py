"""
Unit tests for error handling and validation in Maki framework
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki import Maki
from maki.connector import Connector
from maki.utils import Utils
from maki.agents import Agent, AgentManager

class TestErrorHandling(unittest.TestCase):

    def test_maki_initialization_validation(self):
        """Test that Maki validates initialization parameters"""
        # Test valid initialization
        maki = Maki("localhost", "11434", "llama3", 0.7)
        self.assertEqual(maki.url, "localhost")
        self.assertEqual(maki.port, "11434")
        self.assertEqual(maki.model, "llama3")
        self.assertEqual(maki.temperature, 0.7)

        # Test invalid URL
        with self.assertRaises(ValueError):
            Maki("", "11434", "llama3", 0.7)

        with self.assertRaises(ValueError):
            Maki(None, "11434", "llama3", 0.7)

        # Test invalid port
        with self.assertRaises(ValueError):
            Maki("localhost", "abc", "llama3", 0.7)

        with self.assertRaises(ValueError):
            Maki("localhost", "", "llama3", 0.7)

        # Test invalid model
        with self.assertRaises(ValueError):
            Maki("localhost", "11434", "", 0.7)

        with self.assertRaises(ValueError):
            Maki("localhost", "11434", None, 0.7)

        # Test invalid temperature
        with self.assertRaises(ValueError):
            Maki("localhost", "11434", "llama3", -0.1)

        with self.assertRaises(ValueError):
            Maki("localhost", "11434", "llama3", 1.5)

        with self.assertRaises(ValueError):
            Maki("localhost", "11434", "llama3", "invalid")

    def test_utils_compose_url_validation(self):
        """Test that Utils.compose_url validates parameters"""
        # Test valid call
        result = Utils.compose_url("localhost", "11434", "generate")
        self.assertEqual(result, "http://localhost:11434/api/generate")

        # Test invalid parameters
        with self.assertRaises(ValueError):
            Utils.compose_url("", "11434", "generate")

        with self.assertRaises(ValueError):
            Utils.compose_url("localhost", "", "generate")

        with self.assertRaises(ValueError):
            Utils.compose_url("localhost", "11434", "")

    def test_utils_jsonify_validation(self):
        """Test that Utils.jsonify validates input"""
        # Test valid JSON
        result = Utils.jsonify('{"test": "data"}')
        self.assertEqual(result, {"test": "data"})

        # Test invalid JSON
        with self.assertRaises(ValueError):
            Utils.jsonify("invalid json")

        with self.assertRaises(ValueError):
            Utils.jsonify("")

        with self.assertRaises(ValueError):
            Utils.jsonify(None)

    def test_utils_convert64_validation(self):
        """Test that Utils.convert64 validates input"""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            Utils.convert64("/non/existent/file.jpg")

        # Test invalid file path
        with self.assertRaises(ValueError):
            Utils.convert64("")

        with self.assertRaises(ValueError):
            Utils.convert64(None)

    def test_agent_initialization_validation(self):
        """Test that Agent validates initialization parameters"""
        maki = Maki("localhost", "11434", "llama3", 0.7)

        # Test valid initialization
        agent = Agent("TestAgent", maki, "researcher", "You are a researcher")
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.role, "researcher")
        self.assertEqual(agent.instructions, "You are a researcher")

        # Test invalid name
        with self.assertRaises(ValueError):
            Agent("", maki, "researcher", "You are a researcher")

        with self.assertRaises(ValueError):
            Agent(None, maki, "researcher", "You are a researcher")

    def test_agent_manager_add_agent_validation(self):
        """Test that AgentManager validates add_agent parameters"""
        maki = Maki("localhost", "11434", "llama3", 0.7)
        manager = AgentManager(maki)

        # Test valid addition
        agent = manager.add_agent("TestAgent", "researcher", "You are a researcher")
        self.assertEqual(agent.name, "TestAgent")

        # Test invalid name
        with self.assertRaises(ValueError):
            manager.add_agent("", "researcher", "You are a researcher")

        with self.assertRaises(ValueError):
            manager.add_agent(None, "researcher", "You are a researcher")

        # Test invalid maki_instance
        with self.assertRaises(TypeError):
            manager.add_agent("TestAgent", "researcher", "You are a researcher", "not_a_maki_instance")

if __name__ == '__main__':
    unittest.main()