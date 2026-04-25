"""
Configuration for the Maki Newsletter application.
Edit this file to change the topics, sources, LLM model, and output settings.
"""

# ---------------------------------------------------------------------------
# RSS feed sources — queried directly, no search engine required
# ---------------------------------------------------------------------------

RSS_FEEDS = {
    "TechCrunch":   "https://techcrunch.com/feed/",
    "Wired":        "https://www.wired.com/feed/rss",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "TechRadar":    "https://www.techradar.com/rss",
    "Gizmodo":      "https://gizmodo.com/rss",
    "HackerNews":   "https://news.ycombinator.com/rss",
}

# ---------------------------------------------------------------------------
# Search / ranking
# ---------------------------------------------------------------------------

# Topic queries used to search HackerNews and as keyword hints for RSS
SEARCH_QUERIES = [
    "artificial intelligence latest research 2026",
    "cyber security vulnerabilities",
    "software engineering best practices",
    "cloud infrastructure devops",
    "open source new releases",
    "programming languages updates",
]

# Maximum article candidates before ranking
MAX_CANDIDATES = 30

# Number of articles in the final newsletter
TOP_N = 30

# Max articles fetched per RSS feed
MAX_PER_FEED = 10

# Max HackerNews results per query
MAX_HN_PER_QUERY = 5

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL = "gemma4:26b"
OLLAMA_HOST = "http://localhost:11434"

# Characters sent to the LLM per article (prevents context overflow)
MAX_ARTICLE_CHARS = 10000

# Max words in each article summary
SUMMARY_MAX_WORDS = 300

# ---------------------------------------------------------------------------
# Trend cross-checking
# ---------------------------------------------------------------------------

# Seed keywords sent to Google Trends (max 5 per batch; batched automatically)
TREND_SEED_KEYWORDS = [
    "artificial intelligence",
    "cybersecurity",
    "blockchain",
    "information technology",
    "bitcoin",
    "space",
    "finance",
]

# Google Trends timeframe — "now 7-d" = last 7 days
TREND_TIMEFRAME = "now 7-d"

# Reddit subreddits to scrape for hot posts (no auth required)
REDDIT_SUBREDDITS = [
    "technology",       # IT general
    "artificial",       # AI general
    "MachineLearning",  # AI research
    "netsec",           # Security / offensive
    "cybersecurity",    # Security / defensive
    "ethereum",         # Blockchain
    "Bitcoin",          # Blockchain
]

# Maximum Reddit posts to fetch per subreddit
MAX_REDDIT_PER_SUB = 5

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
ARTICLES_DIR = "output/articles"
