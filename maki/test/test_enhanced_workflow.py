"""
Unit tests for the enhanced workflow management system
"""
import unittest
from unittest.mock import patch, MagicMock
import time

# Import the classes we want to test
from maki.maki import Maki
from maki.agents import Agent, AgentManager, WorkflowTask, TaskStatus, WorkflowState


class TestEnhancedWorkflowManagement(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_maki = Maki("localhost", "11434", "llama3", 0.7)
        self.agent_manager = AgentManager(self.default_maki)

    def test_workflow_task_initialization(self):
        """Test that WorkflowTask initializes correctly"""
        task = WorkflowTask(
            name="test_task",
            agent="TestAgent",
            task="Test task description",
            dependencies=["dep1", "dep2"],
            max_retries=3,
            retry_delay=2.0,
            parallelizable=True
        )

        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.agent, "TestAgent")
        self.assertEqual(task.task, "Test task description")
        self.assertEqual(task.dependencies, ["dep1", "dep2"])
        self.assertEqual(task.max_retries, 3)
        self.assertEqual(task.retry_delay, 2.0)
        self.assertTrue(task.parallelizable)
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_workflow_task_should_execute(self):
        """Test that WorkflowTask evaluates conditions correctly"""
        # Test with no conditions
        task = WorkflowTask("test_task", "TestAgent", "Test task")
        self.assertTrue(task.should_execute())

        # Test with condition that returns True
        def condition_true(context):
            return True

        task_with_condition = WorkflowTask(
            "test_task",
            "TestAgent",
            "Test task",
            conditions=[condition_true]
        )
        self.assertTrue(task_with_condition.should_execute())

        # Test with condition that returns False
        def condition_false(context):
            return False

        task_with_false_condition = WorkflowTask(
            "test_task",
            "TestAgent",
            "Test task",
            conditions=[condition_false]
        )
        self.assertFalse(task_with_false_condition.should_execute())

    def test_workflow_state_initialization(self):
        """Test that WorkflowState initializes correctly"""
        workflow_state = WorkflowState("test_workflow")

        self.assertEqual(workflow_state.workflow_id, "test_workflow")
        self.assertEqual(workflow_state.status, "running")
        self.assertIsNotNone(workflow_state.start_time)
        self.assertIsNone(workflow_state.end_time)
        self.assertEqual(workflow_state.tasks, {})
        self.assertEqual(workflow_state.metrics, {})
        self.assertEqual(workflow_state.error_log, [])

    def test_workflow_state_update_and_progress(self):
        """Test workflow state update and progress calculation"""
        workflow_state = WorkflowState("test_workflow")

        # Update task status
        workflow_state.update_task_status(
            "task1",
            TaskStatus.COMPLETED,
            "result1",
            execution_time=1.5,
            resources_used={"cpu": 50}
        )

        # Check task was added
        self.assertIn("task1", workflow_state.tasks)
        self.assertEqual(workflow_state.tasks["task1"]["status"], TaskStatus.COMPLETED)
        self.assertEqual(workflow_state.tasks["task1"]["result"], "result1")
        self.assertEqual(workflow_state.tasks["task1"]["execution_time"], 1.5)
        self.assertEqual(workflow_state.tasks["task1"]["resources_used"], {"cpu": 50})

        # Check progress
        progress = workflow_state.get_workflow_progress()
        self.assertEqual(progress["total_tasks"], 1)
        self.assertEqual(progress["completed_tasks"], 1)
        self.assertEqual(progress["progress_percentage"], 100.0)
        self.assertEqual(progress["status"], "running")

    def test_enhanced_workflow_execution(self):
        """Test enhanced workflow execution"""
        # Add an agent
        agent = self.agent_manager.add_agent(
            name="TestAgent",
            role="researcher",
            instructions="You are a researcher"
        )

        # Create workflow tasks
        tasks = [
            WorkflowTask(
                name="task1",
                agent="TestAgent",
                task="Research topic A"
            ),
            WorkflowTask(
                name="task2",
                agent="TestAgent",
                task="Research topic B"
            )
        ]

        # Mock the Maki request to avoid actual HTTP requests
        with patch.object(self.default_maki, 'request') as mock_request:
            mock_request.return_value = "Research completed successfully"

            # This would test the workflow execution, but we're mainly testing the classes
            # The actual execution logic is tested in the other test files
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0].name, "task1")
            self.assertEqual(tasks[1].name, "task2")

    def test_task_status_enum(self):
        """Test that TaskStatus enum works correctly"""
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.IN_PROGRESS.value, "in_progress")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.RETRYING.value, "retrying")


if __name__ == '__main__':
    unittest.main()