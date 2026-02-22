"""
Unit tests for Agent and AgentManager functionalities
"""
import unittest
from unittest.mock import patch, MagicMock

# Import the classes we want to test
from maki.maki import Maki
from maki.agents import Agent, AgentManager

class TestAgentFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_maki = Maki("localhost", "11434", "llama3", 0.7)
        self.specialized_maki = Maki("localhost", "11434", "mixtral", 0.3)
        self.agent_manager = AgentManager(self.default_maki)

    def test_agent_initialization(self):
        """Test that Agent initializes correctly"""
        agent = Agent("TestAgent", self.default_maki, "researcher", "You are a researcher")

        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.maki, self.default_maki)
        self.assertEqual(agent.role, "researcher")
        self.assertEqual(agent.instructions, "You are a researcher")

    def test_agent_execute_task(self):
        """Test that Agent can execute tasks"""
        # Mock the Maki request to avoid actual HTTP requests
        with patch.object(self.default_maki, 'request') as mock_request:
            mock_request.return_value = "Test response"

            agent = Agent("TestAgent", self.default_maki, "researcher", "You are a researcher")
            result = agent.execute_task("Test task")

            self.assertEqual(result, "Test response")
            mock_request.assert_called_once()

    def test_agent_memory(self):
        """Test agent memory functionality"""
        agent = Agent("TestAgent", self.default_maki)

        # Test remembering and recalling
        agent.remember("key1", "value1")
        result = agent.recall("key1")
        self.assertEqual(result, "value1")

        # Test non-existent key
        result = agent.recall("nonexistent")
        self.assertIsNone(result)

        # Test clearing memory
        agent.clear_memory()
        result = agent.recall("key1")
        self.assertIsNone(result)

    def test_agent_manager_initialization(self):
        """Test that AgentManager initializes correctly"""
        self.assertEqual(self.agent_manager.maki, self.default_maki)
        self.assertEqual(self.agent_manager.agents, {})

    def test_agent_manager_add_agent_without_maki_instance(self):
        """Test adding agent without specific Maki instance (should use default)"""
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher"
        )

        self.assertIn("TestAgent", self.agent_manager.list_agents())
        self.assertEqual(agent.maki, self.default_maki)  # Should use default Maki

    def test_agent_manager_add_agent_with_maki_instance(self):
        """Test adding agent with specific Maki instance"""
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher",
            maki_instance=self.specialized_maki
        )

        self.assertIn("TestAgent", self.agent_manager.list_agents())
        self.assertEqual(agent.maki, self.specialized_maki)  # Should use specialized Maki

    def test_agent_manager_get_agent(self):
        """Test getting agents from manager"""
        # Add an agent
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher"
        )

        # Retrieve the agent
        retrieved_agent = self.agent_manager.get_agent("TestAgent")
        self.assertEqual(retrieved_agent, agent)

        # Try to get non-existent agent
        retrieved_agent = self.agent_manager.get_agent("NonExistent")
        self.assertIsNone(retrieved_agent)

    def test_agent_manager_remove_agent(self):
        """Test removing agents from manager"""
        # Add an agent
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher"
        )

        # Remove the agent
        self.agent_manager.remove_agent("TestAgent")

        # Verify it's removed
        self.assertNotIn("TestAgent", self.agent_manager.list_agents())
        self.assertIsNone(self.agent_manager.get_agent("TestAgent"))

    def test_agent_manager_list_agents(self):
        """Test listing agents"""
        # Add a few agents
        self.agent_manager.add_agent("Agent1", "researcher", "Research")
        self.agent_manager.add_agent("Agent2", "analyst", "Analyze")

        agents = self.agent_manager.list_agents()
        self.assertEqual(len(agents), 2)
        self.assertIn("Agent1", agents)
        self.assertIn("Agent2", agents)

    def test_agent_manager_assign_task(self):
        """Test assigning tasks to agents"""
        # Add an agent
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher"
        )

        # Mock the Maki request to avoid actual HTTP requests
        with patch.object(self.default_maki, 'request') as mock_request:
            mock_request.return_value = "Task completed successfully"

            result = self.agent_manager.assign_task("TestAgent", "Test task")
            self.assertEqual(result, "Task completed successfully")
            mock_request.assert_called_once()

    def test_backward_compatibility(self):
        """Test that backward compatibility is maintained"""
        # Test that existing usage pattern still works
        manager = AgentManager(self.default_maki)

        # This should work exactly as before (no maki_instance parameter)
        researcher = manager.add_agent(
            name="Researcher",
            role="research analyst",
            instructions="You are an expert researcher"
        )

        # Should use the default Maki instance
        self.assertEqual(researcher.maki, self.default_maki)

        # Should be able to assign tasks
        with patch.object(self.default_maki, 'request') as mock_request:
            mock_request.return_value = "Research completed"
            result = manager.assign_task("Researcher", "Research topic")
            self.assertEqual(result, "Research completed")


if __name__ == '__main__':
    unittest.main()