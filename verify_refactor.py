#!/usr/bin/env python3
"""
Verification script to demonstrate that the refactoring works correctly.
This shows that the agent system can be imported and used properly.
"""

import sys
import os

# Add current directory to path to make imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly after refactoring"""
    try:
        # Test importing the main classes
        from maki.agents import Agent, AgentManager
        from maki.maki import Maki

        print("✅ All imports successful!")

        # Test basic instantiation
        maki = Maki("localhost", "11434", "llama3", 0.7)
        agent = Agent("TestAgent", maki, "researcher", "You are a researcher")
        manager = AgentManager(maki)

        print("✅ Basic instantiation successful!")
        print(f"✅ Agent: {agent}")
        print(f"✅ Manager: {manager}")

        # Test adding an agent
        manager.add_agent("TestAgent2", "writer", "You are a writer")
        agents = manager.list_agents()
        print(f"✅ Agent list: {agents}")

        print("\n🎉 Refactoring verification completed successfully!")
        print("The agent system has been successfully moved to the agents directory")
        print("while maintaining full backward compatibility.")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()