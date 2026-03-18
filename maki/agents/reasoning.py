"""
Reasoning engine for Maki agents.

Provides the ReasoningEngine mixin that gives agents step-by-step reasoning,
self-correction, and task decomposition capabilities.
"""

import functools
import json
import logging
import re
import time
from typing import Dict, List, TYPE_CHECKING

from ..exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError

if TYPE_CHECKING:
    from .protocols import ReasoningHostProtocol

logger = logging.getLogger(__name__)

# Attributes the host class must provide before _init_reasoning() is called.
_REQUIRED_ATTRS = ("maki", "reasoning_history")


def _extract_json_array(text: str) -> str:
    """Extract a JSON array from potentially messy LLM output.

    Handles all common LLM output patterns:
    - Clean JSON arrays
    - Markdown code fences (```json ... ```, ``` ... ```, any language label)
    - Preamble text before the array ("Here is the JSON: [...]")
    - Trailing commentary after the array ("[...] Hope this helps!")
    - Combinations of the above
    """
    # Strip code fence markers with any optional language label
    cleaned = re.sub(r'```[a-zA-Z]*\s*', '', text)
    cleaned = cleaned.strip()

    # Find the outermost JSON array: first '[' to last ']'
    start = cleaned.find('[')
    end = cleaned.rfind(']')
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]

    return cleaned


class ReasoningEngine:
    """
    Mixin that adds reasoning capabilities to an agent.

    **Contract** – the host class must set the following instance attributes
    before ``__init__`` returns:

    * ``maki``              – a Maki LLM backend instance.
    * ``reasoning_history`` – a :class:`collections.deque` for storing steps.

    Enforcement is automatic: :meth:`__init_subclass__` wraps every subclass
    ``__init__`` so that a :exc:`TypeError` is raised immediately after
    construction if any required attribute is missing — even if
    ``super().__init__()`` was never called.

    See :class:`~maki.agents.protocols.ReasoningHostProtocol` for the full
    contract definition.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap the subclass ``__init__`` to enforce the reasoning contract."""
        super().__init_subclass__(**kwargs)
        orig = cls.__dict__.get('__init__')
        if orig is None or getattr(orig, '_reasoning_contract_checked', False):
            return

        @functools.wraps(orig)
        def _checked(self, *args, **kw):
            orig(self, *args, **kw)
            missing = [a for a in _REQUIRED_ATTRS if not hasattr(self, a)]
            if missing:
                raise TypeError(
                    f"'{type(self).__name__}.__init__' completed without setting "
                    f"required ReasoningEngine attribute(s): {missing}. "
                    f"Ensure super().__init__() is called or set these "
                    f"attributes before __init__ returns."
                )

        _checked._reasoning_contract_checked = True
        cls.__init__ = _checked

    def _init_reasoning(self) -> None:
        """
        Validate the ReasoningEngine mixin contract.

        Must be called from the host class ``__init__`` *after* ``maki`` and
        ``reasoning_history`` have been set.

        Raises:
            TypeError: If the host class has not set the required attributes.
        """
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(self, a)]
        if missing:
            raise TypeError(
                f"'{type(self).__name__}' uses ReasoningEngine but is missing "
                f"required attribute(s): {missing}. "
                f"Set these before calling _init_reasoning()."
            )

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
        result = self.maki.request(prompt).content

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
            current = self.maki.request(prompt).content

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
            raw = self.maki.request(prompt)
        except Exception as e:
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            raise MakiNetworkError(f"Failed to get task decomposition from LLM: {str(e)}")

        result = raw.content

        self.reasoning_history.append({
            'original_task': task,
            'decomposition': result,
            'timestamp': time.time()
        })

        json_str = _extract_json_array(result)

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
