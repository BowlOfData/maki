"""
Multi-Agent System for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from typing import Dict, List, Any, Optional
from maki import Maki
import json
import time
import logging


class Agent:
    """An individual agent that can perform tasks using the Maki framework"""

    def __init__(self, name: str, maki_instance: Maki, role: str = "", instructions: str = ""):
        """
        Initialize an agent

        Args:
            name: Unique identifier for the agent
            maki_instance: Maki instance to use for LLM interactions
            role: The role of the agent (e.g., "researcher", "writer", "analyst")
            instructions: Specific instructions for this agent

        Raises:
            ValueError: If name is not a valid string
            TypeError: If maki_instance is not a Maki instance
        """
        logger = logging.getLogger(__name__)

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(role, str):
            raise ValueError("Role must be a string")

        if not isinstance(instructions, str):
            raise ValueError("Instructions must be a string")

        if not isinstance(maki_instance, Maki):
            raise TypeError("maki_instance must be a Maki instance")

        self.name = name.strip()
        self.maki = maki_instance
        self.role = role
        self.instructions = instructions
        self.memory = {}
        self.reasoning_history = []
        self.task_history = []

        logger.info(f"Agent '{self.name}' initialized with role '{self.role}'")

    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"

    def execute_task(self, task: str, context: Optional[Dict] = None) -> str:
        """
        Execute a task using this agent

        Args:
            task: The task to perform
            context: Additional context for the task

        Returns:
            The result of the task execution

        Raises:
            ValueError: If task is not a valid string
            Exception: For HTTP request or other errors
        """
        logger = logging.getLogger(__name__)

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        # Build the prompt with context and instructions
        prompt = f"""
        You are {self.name}, a {self.role}.
        {self.instructions}

        Task: {task}

        Context: {json.dumps(context) if context else 'None'}

        Please provide a detailed response to the task.
        """

        try:
            logger.debug(f"Executing task '{task}' for agent '{self.name}'")
            result = self.maki.request(prompt)
            logger.debug(f"Task '{task}' completed successfully for agent '{self.name}'")
        except Exception as e:
            # Re-raise with more context
            logger.error(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}")
            raise Exception(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}")

        # Record the task execution in history
        self.task_history.append({
            'task': task,
            'context': context,
            'result': result,
            'timestamp': time.time()
        })

        return result

    def remember(self, key: str, value: Any):
        """Store information in the agent's memory"""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory"""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory.clear()

    def think_step_by_step(self, problem: str, steps: int = 3) -> str:
        """Execute reasoning through multiple steps"""
        # Use LLM to break down complex problems into steps
        prompt = f"""
        Break down the following problem into {steps} clear reasoning steps:
        Problem: {problem}

        Provide a structured approach with:
        1. Initial analysis
        2. Key considerations
        3. Solution approach
        """
        result = self.maki.request(prompt)

        # Record the reasoning process
        self.reasoning_history.append({
            'problem': problem,
            'steps': steps,
            'result': result,
            'timestamp': time.time()
        })

        return result

    def self_correct(self, initial_response: str, feedback: str) -> str:
        """Improve response based on feedback"""
        prompt = f"""
        Improve the following response based on feedback:

        Original response: {initial_response}
        Feedback: {feedback}

        Please revise your response to be more accurate and complete.
        """
        result = self.maki.request(prompt)

        # Record the correction process
        self.reasoning_history.append({
            'original_response': initial_response,
            'feedback': feedback,
            'corrected_response': result,
            'timestamp': time.time()
        })

        return result

    def decompose_task(self, task: str, max_subtasks: int = 5) -> List[Dict]:
        """Decompose a complex task into subtasks using LLM.

        Args:
            task: The complex task to decompose
            max_subtasks: Maximum number of subtasks to create

        Returns:
            A list of dictionaries containing subtask details.
            Each dictionary has keys: 'description', 'resources', 'expected_outcome'

        Raises:
            ValueError: If task is not a valid string
            Exception: For LLM request or parsing errors
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
            }},
            {{
                "description": "Design architecture",
                "resources": "Whiteboard, architecture patterns knowledge",
                "expected_outcome": "System diagram and component specifications"
            }}
        ]
        """

        try:
            result = self.maki.request(prompt)
        except Exception as e:
            raise Exception(f"Failed to get task decomposition from LLM: {str(e)}")

        # Record the decomposition process
        self.reasoning_history.append({
            'original_task': task,
            'decomposition': result,
            'timestamp': time.time()
        })

        # Parse the JSON response
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            json_str = result.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            elif json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            subtasks = json.loads(json_str)

            # Validate that it's a list
            if not isinstance(subtasks, list):
                raise ValueError("LLM response is not a JSON array")

            # Validate and normalize each subtask
            validated_subtasks = []
            for i, subtask in enumerate(subtasks[:max_subtasks]):
                if not isinstance(subtask, dict):
                    subtask = {"description": str(subtask)}

                validated_subtasks.append({
                    "description": subtask.get("description", f"Subtask {i+1}"),
                    "resources": subtask.get("resources", "Not specified"),
                    "expected_outcome": subtask.get("expected_outcome", "Not specified")
                })

            return validated_subtasks

        except json.JSONDecodeError as e:
            # If JSON parsing fails, return a fallback with the raw response
            return [{
                "description": f"Task: {task}",
                "resources": "See LLM response",
                "expected_outcome": result[:200] + "..." if len(result) > 200 else result,
                "parsing_error": f"Failed to parse LLM response as JSON: {str(e)}"
            }]
        except Exception as e:
            raise Exception(f"Failed to parse task decomposition: {str(e)}")


class AgentManager:
    """Manages a collection of agents and facilitates their coordination"""

    def __init__(self, maki_instance: Maki):
        """
        Initialize the agent manager

        Args:
            maki_instance: Maki instance to use for LLM interactions
        """
        logger = logging.getLogger(__name__)

        self.maki = maki_instance
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Dict] = []

        logger.info("AgentManager initialized")

    def add_agent(self, name: str, role: str = "", instructions: str = "", maki_instance: Maki = None) -> Agent:
        """
        Add a new agent to the manager

        Args:
            name: Unique identifier for the agent
            role: The role of the agent
            instructions: Specific instructions for this agent
            maki_instance: Optional Maki instance to use for this agent.
                          If not provided, uses the manager's default Maki instance.

        Returns:
            The created Agent instance

        Raises:
            ValueError: If name is not a valid string
            TypeError: If maki_instance is not a Maki instance
        """
        logger = logging.getLogger(__name__)

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(role, str):
            raise ValueError("Role must be a string")

        if not isinstance(instructions, str):
            raise ValueError("Instructions must be a string")

        # Use the provided maki_instance or fall back to the manager's default
        maki_to_use = maki_instance if maki_instance is not None else self.maki

        # Validate that maki_to_use is actually a Maki instance
        if not isinstance(maki_to_use, Maki):
            raise TypeError("maki_instance must be a Maki instance")

        agent = Agent(name, maki_to_use, role, instructions)
        self.agents[name] = agent
        logger.info(f"Added agent '{name}' with role '{role}'")
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get an agent by name

        Args:
            name: The name of the agent to retrieve

        Returns:
            The agent instance or None if not found
        """
        return self.agents.get(name)

    def remove_agent(self, name: str):
        """
        Remove an agent from the manager

        Args:
            name: The name of the agent to remove
        """
        if name in self.agents:
            del self.agents[name]

    def list_agents(self) -> List[str]:
        """
        List all agent names

        Returns:
            A list of agent names
        """
        return list(self.agents.keys())

    def assign_task(self, agent_name: str, task: str, context: Optional[Dict] = None) -> str:
        """
        Assign a task to a specific agent

        Args:
            agent_name: The name of the agent to assign the task to
            task: The task to assign
            context: Additional context for the task

        Returns:
            The result of the task execution

        Raises:
            ValueError: If agent_name or task is not valid
            Exception: For task execution errors
        """
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        try:
            return agent.execute_task(task, context)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Failed to assign task '{task}' to agent '{agent_name}': {str(e)}")

    def coordinate_agents(self, tasks: List[Dict], coordination_prompt: str = "") -> Dict[str, str]:
        """
        Coordinate multiple agents to complete a set of tasks

        Args:
            tasks: List of task dictionaries with 'agent' and 'task' keys
            coordination_prompt: Additional prompt for coordination

        Returns:
            A dictionary mapping task names to results
        """
        results = {}

        # Process tasks in sequence
        for task_dict in tasks:
            agent_name = task_dict.get('agent')
            task = task_dict.get('task')
            context = task_dict.get('context')

            if not agent_name or not task:
                continue

            result = self.assign_task(agent_name, task, context)
            results[task] = result

        return results

    def collaborative_task(self, task: str, agents: List[str], context: Optional[Dict] = None) -> str:
        """
        Have multiple agents work together on a task

        Args:
            task: The main task for collaboration
            agents: List of agent names to participate
            context: Additional context for the task

        Returns:
            A coordinated response from the agents
        """
        # Create a coordination prompt
        prompt = f"""
        You are coordinating a group of agents to solve a task.

        Task: {task}

        Agents involved: {', '.join(agents)}

        Context: {json.dumps(context) if context else 'None'}

        Please provide a coordinated response that synthesizes input from all agents.
        """

        return self.maki.request(prompt)

    def run_workflow(self, workflow: List[Dict]) -> Dict[str, Any]:
        """
        Execute a complete workflow with multiple steps

        Args:
            workflow: List of workflow steps with agent assignments

        Returns:
            Results from all workflow steps
        """
        results = {}

        for step in workflow:
            step_name = step.get('name', f'step_{len(results)}')
            agent_name = step.get('agent')
            task = step.get('task')
            context = step.get('context')

            if agent_name and task:
                result = self.assign_task(agent_name, task, context)
                results[step_name] = {
                    'agent': agent_name,
                    'task': task,
                    'result': result
                }

        return results


# Example usage
if __name__ == "__main__":
    # This would typically be instantiated with a Maki instance
    print("Multi-agent system for Maki framework initialized")