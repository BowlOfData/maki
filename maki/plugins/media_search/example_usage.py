"""
Example usage of the MediaSearch plugin.
"""

import os

from maki.plugins.media_search import MediaSearch


plugin = MediaSearch()
api_key = os.getenv("PEXELS_API_KEY", "")
print(plugin.fetch_pexels_image("artificial intelligence technology", api_key))
