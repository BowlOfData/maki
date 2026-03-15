"""
Reasoning engine for Maki agents.

Provides the ReasoningEngine mixin that gives agents step-by-step reasoning,
self-correction, and task decomposition capabilities.
"""

import json
import logging
import time
from typing import Dict, List

from ..exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Mixin that adds reasoning capabilities to an agent."""

    def think_step_by_step(self, problem: str, steps: int = 3) -> str:
        """Execute reasoning through multiple steps."""
        prompt = f"""
        Break down the following problem into {steps} clear reasoning steps:
        Problem: {problem}

        Provide a structured approach with:
        1. Initial analysis
        2. Key considerations
        3. Solution approach
        """
        result = self.maki.request(prompt)

        self.reasoning_history.append({
            'problem': problem,
            'steps': steps,
            'result': result,
            'timestamp': time.time()
        })

        return result

    def self_correct(self, initial_response: str, feedback: str, max_iterations: int = 1) -> str:
        """
        Iteratively improve a response based on feedback.

        Args:
            initial_response: The response to improve
            feedback: Guidance on how to improve it
            max_iterations: Number of improvement rounds (default 1)

        Returns:
            The improved response after all iterations
        """
        current = initial_response
        for i in range(max_iterations):
            prompt = f"""
            Improve the following response based on feedback:

            Current response: {current}
            Feedback: {feedback}

            Please revise your response to be more accurate and complete.
            """
            current = self.maki.request(prompt)

            self.reasoning_history.append({
                'iteration': i + 1,
                'original_response': initial_response,
                'feedback': feedback,
                'corrected_response': current,
                'timestamp': time.time()
            })

        return current

    def decompose_task(self, task: str, max_subtasks: int = 5) -> List[Dict]:
        """Decompose a complex task into subtasks using LLM.

        Args:
            task: The complex task to decompose
            max_subtasks: Maximum number of subtasks to create

        Returns:
            A list of dicts with keys: 'description', 'resources', 'expected_outcome'

        Raises:
            ValueError: If task is not a valid string or the LLM returns invalid JSON
            MakiNetworkError / MakiTimeoutError / MakiAPIError: For LLM errors
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        prompt = f"""
        Decompose the following task into {max_subtasks} or fewer subtasks.
        Return your response as a JSON array of objects.

        Task: {task}

        For each subtask, provide an object with these exact keys:
        - "description": A clear description of the subtask
        - "resources": Required resources (tools, data, skills needed)
        - "expected_outcome": What successful completion looks like

        Return ONLY the JSON array, no additional text.
        Example format:
        [
            {{
                "description": "Research existing solutions",
                "resources": "Internet access, documentation",
                "expected_outcome": "List of 3-5 comparable solutions with pros/cons"
            }}
        ]
        """

        try:
            result = self.maki.request(prompt)
        except Exception as e:
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            raise MakiNetworkError(f"Failed to get task decomposition from LLM: {str(e)}")

        self.reasoning_history.append({
            'original_task': task,
            'decomposition': result,
            'timestamp': time.time()
        })

        # Strip markdown code fences if present
        json_str = result.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        elif json_str.startswith('```'):
            json_str = json_str[3:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            subtasks = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON for task decomposition: {str(e)}. "
                f"Raw response (first 300 chars): {result[:300]}"
            )

        if not isinstance(subtasks, list):
            raise ValueError("LLM response is not a JSON array")

        validated = []
        for i, subtask in enumerate(subtasks[:max_subtasks]):
            if not isinstance(subtask, dict):
                subtask = {"description": str(subtask)}
            validated.append({
                "description": subtask.get("description", f"Subtask {i + 1}"),
                "resources": subtask.get("resources", "Not specified"),
                "expected_outcome": subtask.get("expected_outcome", "Not specified")
            })

        return validated
