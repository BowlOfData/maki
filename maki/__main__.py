"""
Main module for Maki framework with logging configuration
"""
import logging
import sys

from .config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_PORT,
)

# Configure logging
def configure_logging():
    """Configure logging configuration"""
    # Setup logging configuration with only StreamHandler (no file handler by default)
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format=DEFAULT_LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# Configure logging when module is imported
# Note: This is now commented out to prevent automatic side effects
# configure_logging()

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
    print(
        f"  maki = Maki(url='{DEFAULT_OLLAMA_HOST}', port='{DEFAULT_OLLAMA_PORT}', "
        f"model='{DEFAULT_MODEL}')"
    )

if __name__ == "__main__":
    main()
