#!/usr/bin/env python3
"""
Demonstration of the enhanced reasoning capabilities in the Maki framework
"""

import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki import Maki
from maki.agents import AgentManager

def main():
    print("=== Maki Framework - Enhanced Reasoning Capabilities Demo ===\n")

    # Initialize Maki with a dummy configuration (we won't actually make calls)
    maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)

    # Create an agent manager
    agent_manager = AgentManager(maki)

    # Add an agent with enhanced reasoning capabilities
    researcher = agent_manager.add_agent(
        name="Researcher",
        role="research analyst",
        instructions="You are an expert researcher who can find and analyze information on various topics."
    )

    print("1. Basic Agent Functionality:")
    print(f"   Agent: {researcher.name}")
    print(f"   Role: {researcher.role}")
    print(f"   Instructions: {researcher.instructions}")
    print()

    print("2. Enhanced Reasoning Capabilities:")

    # Test step-by-step thinking
    print("   Step-by-step thinking:")
    print("   - Method: agent.think_step_by_step(problem, steps=3)")
    print("   - Purpose: Break down complex problems into logical steps")
    print("   - Example: How to optimize database queries")
    print("   (In a real implementation, this would call the LLM)")
    print()

    # Test self-correction
    print("   Self-correction:")
    print("   - Method: agent.self_correct(initial_response, feedback)")
    print("   - Purpose: Improve responses based on feedback")
    print("   - Example: Correcting a response about web app performance")
    print("   (In a real implementation, this would call the LLM)")
    print()

    # Test task decomposition
    print("   Task decomposition:")
    print("   - Method: agent.decompose_task(task, max_subtasks=5)")
    print("   - Purpose: Break down complex tasks into manageable subtasks")
    print("   - Example: Developing a complete web application")
    print("   (In a real implementation, this would call the LLM)")
    print()

    print("3. Integration with Multi-Agent System:")
    print("   The enhanced reasoning capabilities work seamlessly with the existing")
    print("   multi-agent framework, allowing for more sophisticated collaboration.")
    print()

    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()