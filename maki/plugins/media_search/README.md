# Media Search Plugin

`media_search` handles asset lookup from external media providers.

## Features

- Pexels landscape image lookup

## Usage

```python
from maki.plugins.media_search import MediaSearch

plugin = MediaSearch()
url = plugin.fetch_pexels_image("technology innovation", api_key="your-key")
print(url)
```
