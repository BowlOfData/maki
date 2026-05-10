"""
Example usage of the JsonReader plugin.
"""

from maki import Maki
from maki.plugins.json_reader import JsonReader


maki = Maki("http://localhost", 11434, "llama3")
plugin = JsonReader(maki)

result = plugin.read_json_fields(
    "data/example.json",
    fields=["title", "summary"],
    max_items=3,
)

print(result)
