"""
Example usage of the JsonReader plugin.
"""

from maki import MakiLLama
from maki.plugins.json_reader import JsonReader


maki = MakiLLama(model="llama3", base_url="http://localhost:11434")
plugin = JsonReader(maki)

result = plugin.read_json_fields(
    "data/example.json",
    fields=["title", "summary"],
    max_items=3,
)

print(result)
