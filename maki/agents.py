"""
Multi-Agent System for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from typing import Dict, List, Any, Optional
from maki.maki import Maki
import json


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
        """
        self.name = name
        self.maki = maki_instance
        self.role = role
        self.instructions = instructions
        self.memory = {}

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
        """
        # Build the prompt with context and instructions
        prompt = f"""
        You are {self.name}, a {self.role}.
        {self.instructions}

        Task: {task}

        Context: {json.dumps(context) if context else 'None'}

        Please provide a detailed response to the task.
        """

        return self.maki.request(prompt)

    def remember(self, key: str, value: Any):
        """Store information in the agent's memory"""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory"""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory.clear()


class AgentManager:
    """Manages a collection of agents and facilitates their coordination"""

    def __init__(self, maki_instance: Maki):
        """
        Initialize the agent manager

        Args:
            maki_instance: Maki instance to use for LLM interactions
        """
        self.maki = maki_instance
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Dict] = []

    def add_agent(self, name: str, role: str = "", instructions: str = "") -> Agent:
        """
        Add a new agent to the manager

        Args:
            name: Unique identifier for the agent
            role: The role of the agent
            instructions: Specific instructions for this agent

        Returns:
            The created Agent instance
        """
        agent = Agent(name, self.maki, role, instructions)
        self.agents[name] = agent
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
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        return agent.execute_task(task, context)

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