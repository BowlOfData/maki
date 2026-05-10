"""
Example usage of the ProviderUpdates plugin.
"""

from maki.plugins.provider_updates import ProviderUpdates


plugin = ProviderUpdates()
print(
    plugin.fetch_model_releases(
        {"OpenAI": "https://openai.com/news/"},
        max_chars=1000,
    )
)
