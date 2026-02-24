"""
Maki - A Python framework for LLM interactions
"""
__version__ = "0.1.0"
__author__ = "Bowl of Data"

# Import main classes for easy access
from .maki import Maki
from .maki.connector import Connector
from .maki.utils import Utils
from .maki.agents.agents import Agent, AgentManager

__all__ = ["Maki", "Connector", "Utils", "Agent", "AgentManager"]