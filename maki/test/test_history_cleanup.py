"""
Unit tests for Agent history cleanup functionality
"""
import unittest
from unittest.mock import patch
from maki.maki import Maki
from maki.agents import Agent

class TestHistoryCleanup(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.maki = Maki("localhost", "11434", "llama3", 0.7)
        self.agent = Agent("TestAgent", self.maki, "researcher", "You are a researcher")

    def test_history_cleanup_initialization(self):
        """Test that agent initializes with empty history"""
        self.assertEqual(len(self.agent.reasoning_history), 0)
        self.assertEqual(len(self.agent.task_history), 0)
        self.assertEqual(self.agent._max_history_entries, 1000)

    def test_history_cleanup_add_task(self):
        """Test that task history gets cleaned up when it exceeds max entries"""
        # Add more entries than the max
        for i in range(1500):
            with patch.object(self.maki, 'request') as mock_request:
                mock_request.return_value = f"Result {i}"
                self.agent.execute_task(f"Task {i}")

        # Check that history was cleaned up to max entries
        self.assertEqual(len(self.agent.task_history), 1000)

        # Check that the oldest entries were removed and newest are kept
        self.assertEqual(self.agent.task_history[0]['task'], "Task 500")  # First entry after cleanup
        self.assertEqual(self.agent.task_history[-1]['task'], "Task 1499")  # Last entry

    def test_history_cleanup_add_reasoning(self):
        """Test that reasoning history gets cleaned up when it exceeds max entries"""
        # Add more entries than the max
        for i in range(1500):
            with patch.object(self.maki, 'request') as mock_request:
                mock_request.return_value = f"Reasoning {i}"
                self.agent.think_step_by_step(f"Problem {i}")

        # Check that history was cleaned up to max entries
        self.assertEqual(len(self.agent.reasoning_history), 1000)

        # Check that the oldest entries were removed and newest are kept
        self.assertEqual(self.agent.reasoning_history[0]['problem'], "Problem 500")  # First entry after cleanup
        self.assertEqual(self.agent.reasoning_history[-1]['problem'], "Problem 1499")  # Last entry

    def test_history_cleanup_set_max_entries(self):
        """Test that max history entries can be set"""
        # Set to a smaller number
        self.agent.set_max_history_entries(500)
        self.assertEqual(self.agent._max_history_entries, 500)

        # Add more entries than the new max
        for i in range(750):
            with patch.object(self.maki, 'request') as mock_request:
                mock_request.return_value = f"Result {i}"
                self.agent.execute_task(f"Task {i}")

        # Check that history was cleaned up to new max entries
        self.assertEqual(len(self.agent.task_history), 500)

        # Check that the oldest entries were removed and newest are kept
        self.assertEqual(self.agent.task_history[0]['task'], "Task 250")  # First entry after cleanup
        self.assertEqual(self.agent.task_history[-1]['task'], "Task 749")  # Last entry

    def test_history_cleanup_with_self_correct(self):
        """Test that self_correct also cleans up history"""
        with patch.object(self.maki, 'request') as mock_request:
            mock_request.return_value = "Corrected response"
            self.agent.self_correct("Original", "Feedback")

        # History should have one entry
        self.assertEqual(len(self.agent.reasoning_history), 1)

        # Add many entries to trigger cleanup
        for i in range(1500):
            with patch.object(self.maki, 'request') as mock_request:
                mock_request.return_value = f"Result {i}"
                self.agent.self_correct(f"Original {i}", f"Feedback {i}")

        # Check that history was cleaned up to max entries
        self.assertEqual(len(self.agent.reasoning_history), 1000)

    def test_history_cleanup_with_decompose_task(self):
        """Test that decompose_task also cleans up history"""
        with patch.object(self.maki, 'request') as mock_request:
            mock_request.return_value = '[{"description": "Subtask 1", "resources": "None", "expected_outcome": "Done"}]'
            self.agent.decompose_task("Main task")

        # History should have one entry
        self.assertEqual(len(self.agent.reasoning_history), 1)

        # Add many entries to trigger cleanup
        for i in range(1500):
            with patch.object(self.maki, 'request') as mock_request:
                mock_request.return_value = '[{"description": "Subtask 1", "resources": "None", "expected_outcome": "Done"}]'
                self.agent.decompose_task(f"Main task {i}")

        # Check that history was cleaned up to max entries
        self.assertEqual(len(self.agent.reasoning_history), 1000)

if __name__ == '__main__':
    unittest.main()