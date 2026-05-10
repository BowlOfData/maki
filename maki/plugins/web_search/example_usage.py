"""Example usage of the WebSearch plugin with the Maki framework."""

from maki.plugins.web_search.web_search import WebSearch

ws = WebSearch()

if __name__ == "__main__":
    print("WebSearch plugin — example usage")
    print("=" * 40)

    # Example 1: RSS feeds filtered by keyword
    print("\nExample 1: RSS feeds — AI articles from this week")
    RSS_FEEDS = {
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "Wired":        "https://www.wired.com/feed/rss",
    }
    articles = ws.search_rss(RSS_FEEDS, max_per_feed=3, keywords=["artificial intelligence", "AI"])
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['url']}")

    # Example 2: HackerNews search
    print("\nExample 2: HackerNews — 'machine learning'")
    articles = ws.search_hackernews("machine learning", max_results=5)
    for a in articles:
        print(f"  {a['title']}")
        print(f"    {a['url']}")

    # Example 3: GitHub Trending repositories (last 7 days)
    print("\nExample 3: GitHub Trending")
    repos = ws.fetch_github_trending(max_results=5)
    for r in repos:
        print(f"  {r['title']}  ({r['snippet'][:60]}…)")
        print(f"    {r['url']}")

    # Example 4: Lobste.rs hot stories
    print("\nExample 4: Lobste.rs hot stories")
    stories = ws.fetch_lobsters(max_results=5)
    for s in stories:
        print(f"  {s['title']}")
        print(f"    {s['url']}")

    # Example 5: Reddit hot posts
    print("\nExample 5: Reddit — r/MachineLearning and r/netsec")
    posts = ws.fetch_reddit_hot(["MachineLearning", "netsec"], max_per_sub=3)
    for p in posts:
        print(f"  [{p['source']}] {p['title']}")
        print(f"    {p['url']}")
