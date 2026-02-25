"""
Main module for Maki framework with logging configuration
"""
import logging
import sys
import os

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'maki.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Setup logging when module is imported
setup_logging()

# Import and expose the main classes
from .maki import Maki
from .agents import Agent, AgentManager

__all__ = ['Maki', 'Agent', 'AgentManager']