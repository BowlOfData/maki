"""
Example usage of the multi-agent system in Maki
"""

# This example shows how to properly use the agent system
# To run this properly, you should run it from the project root directory:
# python -m maki.examples.agent_example

import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki import Maki
from maki.agents import AgentManager, Agent

def main():
    # Initialize Maki with your Ollama instance
    # Replace with your actual Ollama URL and port
    maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)

    # Create an agent manager
    agent_manager = AgentManager(maki)

    # Add some agents
    researcher = agent_manager.add_agent(
        name="Researcher",
        role="research analyst",
        instructions="You are an expert researcher who can find and analyze information on various topics."
    )

    writer = agent_manager.add_agent(
        name="Writer",
        role="content writer",
        instructions="You are a skilled writer who can create clear, well-structured content based on research."
    )

    editor = agent_manager.add_agent(
        name="Editor",
        role="content editor",
        instructions="You are a meticulous editor who can review and improve written content for clarity and correctness."
    )

    print("Available agents:", agent_manager.list_agents())

    # Example 1: Simple task assignment
    print("\n=== Simple Task Assignment ===")
    # Note: This would normally make an actual API call to Ollama
    # For demonstration purposes, we'll show what the call would look like
    print("In a real implementation, this would call the LLM with:")
    print("Prompt: 'Research the benefits of renewable energy'")
    print("Result would be returned from the LLM")

    # Example 2: Demonstrating new reasoning capabilities
    print("\n=== New Reasoning Capabilities ===")

    # Test step-by-step thinking
    print("Testing step-by-step thinking:")
    result = researcher.think_step_by_step("How to improve the performance of a web application?")
    print("Step-by-step analysis result:", result[:100] + "..." if len(result) > 100 else result)

    # Test self-correction
    print("\nTesting self-correction:")
    initial_response = "The web app is slow because of inefficient database queries."
    feedback = "Consider also network latency and client-side rendering issues."
    corrected = researcher.self_correct(initial_response, feedback)
    print("Initial response:", initial_response)
    print("Corrected response:", corrected[:100] + "..." if len(corrected) > 100 else corrected)

    # Test task decomposition
    print("\nTesting task decomposition:")
    complex_task = "Develop a complete web application with user authentication, database integration, and responsive UI"
    subtasks = researcher.decompose_task(complex_task, max_subtasks=3)
    print("Decomposed subtasks:", subtasks)

    # Example 3: Collaborative task
    print("\n=== Collaborative Task ===")
    print("In a real implementation, this would coordinate multiple agents:")
    print("- Researcher would research")
    print("- Writer would write based on research")
    print("- Editor would edit the content")
    print("Result would be a coordinated response from all agents")

    # Example 4: Workflow execution
    print("\n=== Workflow Execution ===")
    print("A workflow would execute multiple steps:")
    print("1. Researcher researches quantum computing developments")
    print("2. Writer summarizes the findings")
    print("3. Editor improves clarity for general audience")
    print("Each step would return results from the LLM")

if __name__ == "__main__":
    main()