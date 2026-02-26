"""
Main module for Maki framework with logging configuration
"""
import logging
import sys
import os

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Setup logging configuration with only StreamHandler (no file handler by default)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# Setup logging when module is imported
# Note: This is now commented out to prevent automatic side effects
# setup_logging()

# Import and expose the main classes
from .maki import Maki
from .agents import Agent, AgentManager

__all__ = ['Maki', 'Agent', 'AgentManager']

def main():
    """Main entry point for the Maki package"""
    print("Maki framework - Multi-agent LLM interactions")
    print("Usage: python -m maki")
    print("")
    print("Available classes:")
    print("  Maki - Main class for interacting with LLMs")
    print("  Agent - Individual agent class")
    print("  AgentManager - Manager for coordinating agents")
    print("")
    print("Example usage:")
    print("  from maki import Maki")
    print("  maki = Maki(url='http://localhost', port='11434', model='qwen3-coder:30b')")

if __name__ == "__main__":
    main()