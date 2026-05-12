"""
Unit tests for error handling and validation in Maki framework
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki.connector import Connector
from maki.utils import Utils
from maki.agents import Agent, AgentManager

class TestErrorHandling(unittest.TestCase):

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
        # Test with non-existent file - we need to make the path within working directory
        # but simulate that the file doesn't exist by patching os.path.exists
        with patch('maki.utils.os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                Utils.convert64("non_existent_file.jpg")

        # Test invalid file path
        with self.assertRaises(ValueError):
            Utils.convert64("")

        with self.assertRaises(ValueError):
            Utils.convert64(None)

    def test_agent_initialization_validation(self):
        """Test that Agent validates initialization parameters"""
        maki = MagicMock()

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
        maki = MagicMock()
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
