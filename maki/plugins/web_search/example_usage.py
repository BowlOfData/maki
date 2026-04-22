"""
Example usage of the WebSearch plugin with the Maki framework.
"""

from maki.plugins.web_search.web_search import WebSearch

# Initialise the plugin (no Maki LLM instance required for search)
ws = WebSearch()

if __name__ == "__main__":
    print("WebSearch plugin — example usage")
    print("=" * 40)

    # Example 1: DuckDuckGo news search
    print("\nExample 1: DuckDuckGo news — 'AI research'")
    articles = ws.search_news("AI research", max_results=5, time_filter="w")
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['url']}")

    # Example 2: Site-scoped search
    print("\nExample 2: Site-scoped — TechCrunch AI news")
    articles = ws.search_news("AI site:techcrunch.com", max_results=3, time_filter="w")
    for a in articles:
        print(f"  {a['title']}")
        print(f"    {a['url']}")

    # Example 3: Multi-query deduplicated search
    print("\nExample 3: Multi-query search")
    articles = ws.search_articles(
        queries=["cybersecurity vulnerabilities", "open source releases"],
        max_per_query=3,
    )
    print(f"  {len(articles)} unique articles found")

    # Example 4: HackerNews search
    print("\nExample 4: HackerNews — 'machine learning'")
    articles = ws.search_hackernews("machine learning", max_results=5)
    for a in articles:
        print(f"  {a['title']}")
        print(f"    {a['url']}")
