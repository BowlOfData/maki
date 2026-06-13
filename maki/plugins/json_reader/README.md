# JSON Reader Plugin

`json_reader` reads JSON array files and returns only the requested fields in a compact text format that is easy to pass into agent prompts.

## Features

- Restricts file access to a configurable base directory
- Reads top-level JSON arrays
- Selects only the fields you ask for
- Limits the number of returned items when needed

## Example

```python
from maki import MakiLLama
from maki.plugins.json_reader import JsonReader

llm = MakiLLama()
plugin = JsonReader(llm)

result = plugin.read_json_fields(
    "data/articles.json",
    fields=["title", "author", "tags"],
    max_items=5,
)

print(result["content"])
```
