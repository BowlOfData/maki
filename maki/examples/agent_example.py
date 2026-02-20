"""
Example usage of the multi-agent system in Maki
"""

from maki.maki import Maki
from maki.agents import AgentManager

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
    result = agent_manager.assign_task("Researcher", "Research the benefits of renewable energy")
    print(f"Researcher result: {result}")

    # Example 2: Collaborative task
    print("\n=== Collaborative Task ===")
    result = agent_manager.collaborative_task(
        task="Write an article about AI ethics",
        agents=["Researcher", "Writer", "Editor"]
    )
    print(f"Collaborative result: {result}")

    # Example 3: Workflow execution
    print("\n=== Workflow Execution ===")
    workflow = [
        {
            "name": "research",
            "agent": "Researcher",
            "task": "Research the latest developments in quantum computing",
            "context": {"focus": "applications"}
        },
        {
            "name": "write",
            "agent": "Writer",
            "task": "Write a summary of the quantum computing research findings",
            "context": {"tone": "technical"}
        },
        {
            "name": "edit",
            "agent": "Editor",
            "task": "Edit and improve the technical summary for clarity",
            "context": {"audience": "general public"}
        }
    ]

    results = agent_manager.run_workflow(workflow)
    for step_name, step_result in results.items():
        print(f"{step_name}: {step_result['result'][:100]}...")

if __name__ == "__main__":
    main()