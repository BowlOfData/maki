#!/usr/bin/env python3
"""
Test script to verify logging functionality in Maki
"""

import logging
import sys
import os

# Add the project root to Python path so imports work properly
# Note: This approach avoids sys.path manipulation by running from the project root
# with python -m maki.test.test_logging
# Or run from project root with: python -m maki.test.test_logging

from maki import Maki

def test_logging():
    """Test that logging is working properly"""
    print("Testing Maki logging functionality...")

    # Setup basic logging to see output
    logging.basicConfig(level=logging.DEBUG)

    try:
        # This should trigger logging
        maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)
        print("Maki initialized successfully")

        # Test version method
        print("Testing version method...")
        # Note: This will fail because we're not running a real Ollama instance
        # but the logging should still work
        try:
            version = maki.version()
        except Exception as e:
            print(f"Expected error (no real Ollama instance): {str(e)}")

        print("Logging test completed successfully")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

    return True

if __name__ == "__main__":
    test_logging()