"""
Test script to verify the reasoning enhancements in the Agent class
"""

import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki import Maki
from maki.agents import Agent

def test_reasoning_enhancements():
    # Initialize Maki with a dummy configuration (we won't actually make calls)
    maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)

    # Create an agent
    agent = Agent(name="TestAgent", maki_instance=maki, role="tester", instructions="You are a testing agent")

    print("Testing Agent reasoning enhancements...")

    # Test 1: Check that basic agent functionality still works
    print("\n1. Testing basic agent functionality:")
    print(f"Agent name: {agent.name}")
    print(f"Agent role: {agent.role}")

    # Test 2: Test reasoning history tracking
    print("\n2. Testing reasoning history tracking:")
    print(f"Initial reasoning history length: {len(agent.reasoning_history)}")

    # Test 3: Test step-by-step thinking
    print("\n3. Testing step-by-step thinking:")
    try:
        result = agent.think_step_by_step("How to optimize database queries?")
        print("Step-by-step thinking result:", result[:50] + "..." if len(result) > 50 else result)
        print(f"Reasoning history length after thinking: {len(agent.reasoning_history)}")
    except Exception as e:
        print(f"Error in step-by-step thinking: {e}")

    # Test 4: Test self-correction
    print("\n4. Testing self-correction:")
    try:
        initial = "Database queries are slow due to lack of indexing."
        feedback = "Also consider query execution plans and connection pooling."
        corrected = agent.self_correct(initial, feedback)
        print("Initial response:", initial)
        print("Corrected response:", corrected[:50] + "..." if len(corrected) > 50 else corrected)
        print(f"Reasoning history length after correction: {len(agent.reasoning_history)}")
    except Exception as e:
        print(f"Error in self-correction: {e}")

    # Test 5: Test task decomposition
    print("\n5. Testing task decomposition:")
    try:
        task = "Build a complete web application with authentication"
        subtasks = agent.decompose_task(task, max_subtasks=3)
        print("Decomposed subtasks:", subtasks)
        print(f"Reasoning history length after decomposition: {len(agent.reasoning_history)}")
    except Exception as e:
        print(f"Error in task decomposition: {e}")

    print("\nTesting completed!")

if __name__ == "__main__":
    test_reasoning_enhancements()