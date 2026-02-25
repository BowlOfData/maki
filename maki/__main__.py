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