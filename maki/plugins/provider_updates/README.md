# Provider Updates Plugin

`provider_updates` is responsible for scraping provider announcement pages and returning cleaned release text.

## Features

- Fetches provider news/update pages
- Strips HTML into compact text
- Handles noisy loading-state pages

## Usage

```python
from maki.plugins.provider_updates import ProviderUpdates

plugin = ProviderUpdates()
items = plugin.fetch_model_releases(
    {"OpenAI": "https://openai.com/news/"}
)
print(items)
```
