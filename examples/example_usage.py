"""
Example usage of the WebToMd plugin with Maki agents
"""

import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki.maki import Maki
from maki.plugins.web_to_md.web_to_md import WebToMd

# Initialize Maki
maki = Maki("localhost", 11434, "qwen3-coder:30b", temperature=0)

# Initialize the web_to_md plugin
web_to_md = WebToMd(maki)

# Example: Fetch a web page and convert to markdown
def fetch_and_save_webpage(url, output_file=None):
    """
    Fetch a webpage and save it as markdown.

    Args:
        url (str): The URL to fetch
        output_file (str, optional): Output filename. If None, generates one.

    Returns:
        str: Result message
    """
    result = web_to_md.fetch_and_convert_to_md(url, output_file)

    if result['success']:
        return f"Successfully fetched and saved {url} to {result['output_file']}"
    else:
        return f"Failed to fetch {url}: {result['error']}"

if __name__ == "__main__":
    print("WebToMd plugin example usage")
    print("================================")

    # Example 1: Basic usage
    print("Example 1: Fetching a simple page")
    result = fetch_and_save_webpage("https://www.repubblica.it/")
    print(result)

    # Example 2: With custom output file
    print("\nExample 2: Fetching with custom output file")
    result = fetch_and_save_webpage("https://www.repubblica.it/", "test_page.md")
    print(result)

    print("\nPlugin is ready to be used with Maki agents")