"""
Newsletter pipeline — six dedicated stages, each implemented as an agent
or a dedicated function as required by the plan.

Stages
------
1. search_stage()       — dedicated function  — finds article candidates
2. download_stage()     — dedicated function  — downloads articles as Markdown
3. reader_agent         — LLM agent           — extracts metadata from each article
4. ranker_agent         — LLM agent           — ranks articles, selects top N
5. summarizer_agent     — LLM agent           — writes ≤200-word summaries
6. writer_agent         — LLM agent           — assembles the final newsletter

Data is passed explicitly between stages: each stage returns a Python
object that is passed directly to the next stage.  For LLM agents the
data is serialised as JSON inside the task/context strings.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from maki.agents.agent_manager import AgentManager
from maki.makiLLama import MakiLLama
from maki.plugins.file_reader.file_reader import FileReader
from maki.plugins.file_writer.file_writer import FileWriter
from maki.plugins.web_to_md.web_to_md import WebToMd

from .config import (
    ARTICLES_DIR,
    LLM_MODEL,
    MAX_ARTICLE_CHARS,
    MAX_CANDIDATES,
    MAX_HN_PER_QUERY,
    MAX_PER_FEED,
    MAX_REDDIT_PER_SUB,
    OLLAMA_HOST,
    OUTPUT_DIR,
    REDDIT_SUBREDDITS,
    RSS_FEEDS,
    SEARCH_QUERIES,
    SUMMARY_MAX_WORDS,
    TOP_N,
    TREND_SEED_KEYWORDS,
    TREND_TIMEFRAME,
)
from maki.plugins.web_search.web_search import WebSearch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(url: str) -> str:
    """Convert a URL to a safe filename stem (max 80 chars)."""
    parsed = urlparse(url)
    raw = (parsed.netloc + parsed.path).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return slug[:80] or "article"


def _extract_json(text: str) -> Any:
    """
    Robustly extract the first JSON value (object or array) from an LLM response.
    Returns the parsed value, or None on failure.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try the whole cleaned string first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first [ or { and attempt progressively longer substrings
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = cleaned.find(start_char)
        if idx == -1:
            continue
        # Walk backwards from end to find the matching closer
        end_idx = cleaned.rfind(end_char)
        if end_idx == -1 or end_idx <= idx:
            continue
        try:
            return json.loads(cleaned[idx: end_idx + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("_extract_json: could not parse JSON from LLM response")
    return None


def _truncate(content: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Return the first max_chars characters of a string."""
    return content[:max_chars] if len(content) > max_chars else content


# ---------------------------------------------------------------------------
# NewsletterPipeline
# ---------------------------------------------------------------------------

class NewsletterPipeline:
    """
    Orchestrates the six newsletter stages using AgentManager and dedicated
    functions for the non-LLM stages.
    """

    def __init__(self) -> None:
        logger.info("Initialising LLM backend: %s @ %s", LLM_MODEL, OLLAMA_HOST)
        self.llm = MakiLLama(model=LLM_MODEL, base_url=OLLAMA_HOST)

        # AgentManager creates and holds all LLM agents
        self.manager = AgentManager(self.llm)
        self._register_agents()

        # Plugins used directly by the non-LLM stages
        self.web_search = WebSearch()

        # WebToMd writes relative to CWD (set to maki_newsletter/ in main.py)
        self.web_to_md = WebToMd()

        # FileReader/FileWriter scoped to the articles and output directories
        self.file_reader = FileReader(base_dir=os.path.abspath(ARTICLES_DIR))
        self.file_writer = FileWriter(base_dir=os.path.abspath(OUTPUT_DIR))

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def _register_agents(self) -> None:
        self.manager.add_agent(
            name="reader_agent",
            role="technical content analyst",
            instructions=(
                "You analyse tech article content and extract structured metadata. "
                "You always respond with a valid JSON object and nothing else. "
                "Do NOT include markdown fences or explanatory text."
            ),
        )
        self.manager.add_agent(
            name="ranker_agent",
            role="technical relevance judge",
            instructions=(
                "You rank tech articles by relevance, novelty, and technical depth. "
                "You always respond with a valid JSON array and nothing else. "
                "Do NOT include markdown fences or explanatory text."
            ),
        )
        self.manager.add_agent(
            name="summarizer_agent",
            role="technical writer",
            instructions=(
                f"You write concise technical summaries of no more than {SUMMARY_MAX_WORDS} words. "
                "Focus on key insights, technologies mentioned, and practical impact. "
                "Return only the summary text — no titles, headings, or preamble."
            ),
        )
        self.manager.add_agent(
            name="writer_agent",
            role="newsletter editor",
            instructions=(
                "You assemble polished technical newsletters. "
                "You write clear, engaging introductions that synthesise themes "
                "across multiple articles."
            ),
        )

    # ------------------------------------------------------------------
    # Stage 1 — Search (dedicated function)
    # ------------------------------------------------------------------

    def stage_search(self) -> List[Dict[str, Any]]:
        """
        Fetch current-week articles from RSS feeds and HackerNews.
        Returns up to MAX_CANDIDATES deduplicated article dicts.
        """
        logger.info("Stage 1: fetching articles from RSS feeds and HackerNews …")
        seen_urls: set = set()
        candidates: List[Dict[str, Any]] = []

        def _add(articles: List[Dict]) -> None:
            for a in articles:
                url = a.get("url", "")
                if url and url not in seen_urls and len(candidates) < MAX_CANDIDATES:
                    seen_urls.add(url)
                    candidates.append(a)

        # Primary: RSS feeds queried directly (no search engine)
        _add(self.web_search.search_rss(RSS_FEEDS, max_per_feed=MAX_PER_FEED))

        # Secondary: HackerNews Algolia API for each topic query
        for query in SEARCH_QUERIES:
            if len(candidates) >= MAX_CANDIDATES:
                break
            _add(self.web_search.search_hackernews(query, max_results=MAX_HN_PER_QUERY))

        logger.info("Stage 1 complete: %d candidates found", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Stage 1b — Trends (dedicated function)
    # ------------------------------------------------------------------

    def stage_trends(self) -> tuple:
        """
        Fetch weekly trend signals from Google Trends and Reddit.

        Returns a tuple of:
          - trend_articles (List[Dict]): hot Reddit posts to merge into candidates
          - trending_keywords (List[str]): flat list of rising query strings from
            Google Trends + Reddit post titles, used to boost article ranking
        """
        logger.info("Stage 1b: fetching trend signals from Google Trends and Reddit …")

        # Run both fetches; failures are logged internally and return empty results
        google_trends = self.web_search.fetch_google_trends(
            TREND_SEED_KEYWORDS, timeframe=TREND_TIMEFRAME
        )
        reddit_articles = self.web_search.fetch_reddit_hot(
            REDDIT_SUBREDDITS, max_per_sub=MAX_REDDIT_PER_SUB
        )

        # Flatten Google Trends rising queries into a single keyword list
        trending_keywords: list = []
        for seed, queries in google_trends.items():
            for q in queries:
                if q and q not in trending_keywords:
                    trending_keywords.append(q)

        # Also extract keywords from Reddit post titles (first 6 words each)
        for post in reddit_articles:
            title_words = post.get("title", "").split()[:6]
            phrase = " ".join(title_words)
            if phrase and phrase not in trending_keywords:
                trending_keywords.append(phrase)

        logger.info(
            "Stage 1b complete: %d trending keywords, %d Reddit articles",
            len(trending_keywords), len(reddit_articles),
        )
        return reddit_articles, trending_keywords

    # ------------------------------------------------------------------
    # Stage 2 — Download (dedicated function)
    # ------------------------------------------------------------------

    def stage_download(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Download each candidate article as Markdown.
        Returns a list of successful download dicts (adds 'local_path' key).
        """
        logger.info("Stage 2: downloading %d articles …", len(candidates))
        downloaded: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        month_name = now.strftime("%B").lower()     # e.g. "april"
        week_num = now.isocalendar()[1]             # e.g. 14
        articles_abs = os.path.abspath(
            os.path.join(ARTICLES_DIR, month_name, str(week_num))
        )
        os.makedirs(articles_abs, exist_ok=True)

        # Store for later stages
        self._articles_week_dir = articles_abs
        self._week_num = week_num
        self._year = now.year

        for article in candidates:
            url = article.get("url", "")
            if not url:
                continue

            filename = _slug(url) + ".md"
            output_path = os.path.join(articles_abs, filename)

            if os.path.exists(output_path):
                logger.debug("Already exists, skipping download: %s", filename)
                downloaded.append(
                    {
                        **article,
                        "local_path": output_path,
                        "filename": filename,
                    }
                )
                continue

            result = self.web_to_md.fetch_and_convert_to_md(url, output_path)

            if result["success"]:
                content_len = len(result.get("content", ""))
                if content_len < 300:
                    logger.debug("Skipping %s — content too short (%d chars)", url, content_len)
                    continue
                downloaded.append(
                    {
                        **article,
                        "local_path": output_path,
                        "filename": filename,
                    }
                )
                logger.debug("Downloaded: %s → %s", url, filename)
            else:
                logger.warning("Failed to download %s: %s", url, result.get("error"))

            # Polite delay to reduce rate-limit pressure
            time.sleep(1.5)

        logger.info("Stage 2 complete: %d articles downloaded", len(downloaded))
        return downloaded

    # ------------------------------------------------------------------
    # Stage 3 — Read (LLM agent)
    # ------------------------------------------------------------------

    def stage_read(
        self, downloaded: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use reader_agent to extract structured metadata from each article.
        Returns an enriched list (adds main_topic, key_points, technologies, quality_score).
        """
        logger.info("Stage 3: analysing %d articles …", len(downloaded))
        agent = self.manager.get_agent("reader_agent")
        enriched: List[Dict[str, Any]] = []

        for article in downloaded:
            local_path = article.get("local_path", "")
            meta_path = local_path.replace(".md", "_meta.json") if local_path else ""

            # Load cached metadata if already evaluated
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        metadata = json.load(fh)
                    logger.debug("Loaded cached metadata: %s", meta_path)
                    enriched.append({**article, **metadata})
                    continue
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not load cached metadata %s: %s — re-evaluating", meta_path, exc)

            try:
                raw = open(local_path, encoding="utf-8").read()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", local_path, exc)
                continue

            content = _truncate(raw)
            task = (
                "Analyse the following technical article and return a JSON object with "
                "exactly these keys:\n"
                '  "main_topic": string (one sentence),\n'
                '  "key_points": array of up to 5 strings,\n'
                '  "technologies": array of technology/product names mentioned,\n'
                '  "quality_score": integer 0-10 (technical depth and relevance).\n\n'
                "Article content:\n"
                f"{content}"
            )

            try:
                raw_response = agent.execute_task(task)
                metadata = _extract_json(raw_response)
            except Exception as exc:
                logger.warning("reader_agent failed for %s: %s", article.get("url"), exc)
                metadata = None

            if not isinstance(metadata, dict):
                logger.warning(
                    "Could not parse JSON from LLM response for %s — removing article files",
                    article.get("url"),
                )
                for path in (local_path, meta_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.debug("Removed: %s", path)
                        except OSError as exc:
                            logger.warning("Could not remove %s: %s", path, exc)
                continue

            # Discard low-quality articles immediately
            if int(metadata.get("quality_score", 0)) < 4:
                logger.debug(
                    "Score %s < 4 for %s — removing article files",
                    metadata.get("quality_score"), article.get("url"),
                )
                for path in (local_path, meta_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError as exc:
                            logger.warning("Could not remove %s: %s", path, exc)
                continue

            # Persist metadata so subsequent runs skip the LLM call
            if meta_path:
                try:
                    with open(meta_path, "w", encoding="utf-8") as fh:
                        json.dump(metadata, fh, indent=2, ensure_ascii=False)
                except OSError as exc:
                    logger.warning("Could not save metadata %s: %s", meta_path, exc)

            enriched.append({**article, **metadata})

        logger.info("Stage 3 complete: %d articles enriched", len(enriched))
        return enriched

    # ------------------------------------------------------------------
    # Stage 4 — Rank (LLM agent)
    # ------------------------------------------------------------------

    def stage_rank(
        self,
        enriched: List[Dict[str, Any]],
        trending_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use ranker_agent to rank all articles and select the top TOP_N.

        When trending_keywords is provided (from stage_trends), the LLM is
        instructed to boost articles that cover those topics.

        Returns a sorted list of the top TOP_N article dicts.
        """
        logger.info("Stage 4: ranking %d articles …", len(enriched))
        agent = self.manager.get_agent("ranker_agent")

        # Build a compact summary for each article to stay within the context window
        compact = [
            {
                "index": i,
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "main_topic": a.get("main_topic", ""),
                "key_points": a.get("key_points", []),
                "technologies": a.get("technologies", []),
                "quality_score": a.get("quality_score", 5),
            }
            for i, a in enumerate(enriched)
        ]

        trend_context = ""
        if trending_keywords:
            # Limit to the 40 most relevant keywords to avoid context bloat
            kw_sample = trending_keywords[:40]
            trend_context = (
                "\n\nTrending topics this week (from Google Trends + Reddit) — "
                "prioritise articles that directly cover these:\n"
                + ", ".join(f'"{kw}"' for kw in kw_sample)
                + "\n"
            )

        task = (
            f"You are given {len(compact)} tech articles. "
            f"Rank them by relevance, novelty, and technical depth. "
            f"Select the top {TOP_N} most valuable articles for a technical audience. "
            "Return a JSON array of the selected article indexes (integers) in ranked order "
            f"(best first), containing exactly {TOP_N} elements."
            f"{trend_context}\n\n"
            "Articles:\n"
            f"{json.dumps(compact, indent=2)}"
        )

        top10: List[Dict[str, Any]] = []
        try:
            raw_response = agent.execute_task(task)
            indexes = _extract_json(raw_response)

            if isinstance(indexes, list):
                # Some models return list of dicts; extract the index field
                clean_indexes = []
                for item in indexes:
                    if isinstance(item, int):
                        clean_indexes.append(item)
                    elif isinstance(item, dict):
                        for key in ("index", "id", "rank"):
                            if key in item:
                                clean_indexes.append(int(item[key]))
                                break
                indexes = clean_indexes

            if isinstance(indexes, list) and indexes:
                for idx in indexes[: TOP_N]:
                    if 0 <= idx < len(enriched):
                        top10.append(enriched[idx])
        except Exception as exc:
            logger.warning("ranker_agent failed: %s — using quality_score fallback", exc)

        # Fallback: sort by quality_score if LLM ranking failed or returned too few
        if len(top10) < TOP_N:
            logger.warning(
                "Ranking returned %d articles (expected %d); filling with quality_score fallback",
                len(top10),
                TOP_N,
            )
            ranked_by_score = sorted(
                enriched, key=lambda a: a.get("quality_score", 0), reverse=True
            )
            existing_urls = {a["url"] for a in top10}
            for a in ranked_by_score:
                if a["url"] not in existing_urls and len(top10) < TOP_N:
                    top10.append(a)

        logger.info("Stage 4 complete: %d articles selected", len(top10))
        return top10[:TOP_N]

    # ------------------------------------------------------------------
    # Stage 5 — Summarize (LLM agent)
    # ------------------------------------------------------------------

    def stage_summarize(
        self, top10: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use summarizer_agent to produce a ≤200-word summary for each top article.
        Returns the list with a 'summary' key added.
        """
        logger.info("Stage 5: summarising %d articles …", len(top10))
        agent = self.manager.get_agent("summarizer_agent")
        summaries: List[Dict[str, Any]] = []

        for article in top10:
            local_path = article.get("local_path", "")
            try:
                raw = open(local_path, encoding="utf-8").read()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", local_path, exc)
                raw = article.get("snippet", "")

            content = _truncate(raw)
            task = (
                f"Write a technical summary of no more than {SUMMARY_MAX_WORDS} words "
                "for the following article. Focus on what is new, what technology is involved, "
                "and why it matters to software engineers.\n\n"
                f"Title: {article.get('title', '')}\n\n"
                f"Content:\n{content}"
            )

            summary = ""
            try:
                summary = agent.execute_task(task).strip()
            except Exception as exc:
                logger.warning("summarizer_agent failed for %s: %s", article.get("url"), exc)
                summary = article.get("snippet", "No summary available.")

            summaries.append({**article, "summary": summary})

        logger.info("Stage 5 complete: %d summaries written", len(summaries))
        return summaries

    # ------------------------------------------------------------------
    # Stage 6 — Evaluate (write evaluation file + stop)
    # ------------------------------------------------------------------

    def stage_evaluate(
        self,
        summaries: List[Dict[str, Any]],
        trending_keywords: Optional[List[str]] = None,
    ) -> str:
        """
        Write an evaluation file grouping articles by quality score:
          0-3  → Discard
          4-5  → Needs review
          6-10 → Good match

        Saves:
          • evaluate_<week>.md  in the articles week directory
          • summaries_<week>_<year>.json  in OUTPUT_DIR (used by generate.py)

        Returns the absolute path of the evaluation Markdown file.
        """
        logger.info("Stage 6: writing evaluation file …")

        week_num = getattr(self, "_week_num", datetime.now(timezone.utc).isocalendar()[1])
        year = getattr(self, "_year", datetime.now(timezone.utc).year)
        articles_week_dir = getattr(
            self, "_articles_week_dir",
            os.path.abspath(ARTICLES_DIR),
        )
        now = datetime.now(timezone.utc)

        # Pre-compute lowercase trend keywords for fast matching
        kw_lower = [kw.lower() for kw in (trending_keywords or [])]

        def _matched_trends(article: Dict) -> List[str]:
            """Return trend keywords that appear in the article's title or main_topic."""
            haystack = (
                article.get("title", "") + " " + article.get("main_topic", "")
            ).lower()
            return [kw for kw in kw_lower if kw in haystack]

        # ---------- bucket by score ----------
        good: List[Dict] = []
        review: List[Dict] = []
        discard: List[Dict] = []
        for a in summaries:
            score = int(a.get("quality_score", 5))
            if score >= 6:
                good.append(a)
            elif score >= 4:
                review.append(a)
            else:
                discard.append(a)

        def _article_block(rank: int, a: Dict) -> List[str]:
            title = a.get("title", f"Article {rank}")
            url = a.get("url", "")
            source = a.get("source", urlparse(url).netloc if url else "")
            published = a.get("published", "")
            score = a.get("quality_score", "?")
            summary = a.get("summary", "")
            pub_str = f" | **Published:** {published}" if published else ""
            matched = _matched_trends(a)
            trend_str = (
                f"\n**Trending topics matched:** {', '.join(matched)}" if matched else ""
            )
            return [
                f"### {rank}. {title}",
                f"**Source:** {source}{pub_str} | **Score:** {score}/10",
                f"**URL:** {url}{trend_str}",
                "",
                summary,
                "",
                "---",
                "",
            ]

        trend_section: List[str] = []
        if kw_lower:
            trend_section = [
                "### Trending topics this week",
                ", ".join(trending_keywords[:40]),
                "",
            ]

        lines: List[str] = [
            f"# Newsletter Evaluation — Week {week_num}, {year}",
            f"*Generated on {now.strftime('%Y-%m-%d')}*",
            "",
            f"**Summary:** {len(good)} good match · {len(review)} needs review · {len(discard)} discard",
            "",
            *trend_section,
            "---",
            "",
            "## ✅ Good Match (score 6–10) — will be included in newsletter",
            "",
        ]
        for i, a in enumerate(good, 1):
            lines += _article_block(i, a)

        lines += [
            "## ⚠️ Needs Review (score 4–5) — check before including",
            "",
        ]
        for i, a in enumerate(review, 1):
            lines += _article_block(i, a)

        lines += [
            "## ❌ Discard (score 0–3) — low relevance / off-topic",
            "",
        ]
        for i, a in enumerate(discard, 1):
            lines += _article_block(i, a)

        eval_md = "\n".join(lines)
        eval_filename = f"evaluate_{week_num}.md"
        eval_path = os.path.join(articles_week_dir, eval_filename)
        os.makedirs(articles_week_dir, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as fh:
            fh.write(eval_md)
        logger.info("Evaluation file written to %s", eval_path)

        # ---------- persist summaries JSON for generate.py ----------
        json_filename = f"summaries_{week_num:02d}_{year}.json"
        json_path = os.path.join(os.path.abspath(OUTPUT_DIR), json_filename)
        os.makedirs(os.path.abspath(OUTPUT_DIR), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summaries, fh, indent=2, ensure_ascii=False)
        logger.info("Summaries JSON written to %s", json_path)

        logger.info("Stage 6 complete.")
        return eval_path

    # ------------------------------------------------------------------
    # Stage 7 — Write (LLM agent + file_writer)
    # ------------------------------------------------------------------

    def stage_write(self, summaries: List[Dict[str, Any]]) -> str:
        """
        Assemble and write the final newsletter Markdown file.
        Returns the absolute path of the output file.
        """
        logger.info("Stage 7: assembling newsletter …")
        agent = self.manager.get_agent("writer_agent")

        now = datetime.now(timezone.utc)
        week_num = getattr(self, "_week_num", now.isocalendar()[1])
        year = getattr(self, "_year", now.year)

        # Ask the writer agent for an editorial introduction
        intro_task = (
            "Write a short editorial introduction (10-15 sentences) for a weekly tech newsletter. "
            "The introduction should synthesise the main themes across the articles listed below "
            "and give readers an overview of what to expect.\n\n"
            "Articles this week:\n"
            + "\n".join(
                f"- {a.get('title', '')} ({a.get('source', '')})"
                for a in summaries
            )
        )
        try:
            introduction = agent.execute_task(intro_task).strip()
        except Exception as exc:
            logger.warning("writer_agent intro failed: %s", exc)
            introduction = (
                f"Welcome to Week {week_num} of the tech newsletter. "
                "Here are the top stories from the past week."
            )

        # Assemble Markdown
        lines: List[str] = [
            f"# Bowl of Data - Tech Newsletter — Week {week_num}, {year}",
            f"*Generated on {now.strftime('%Y-%m-%d')}*",
            "",
            "---",
            "",
            "## Introduction",
            "",
            introduction,
            "",
            "---",
            "",
        ]

        for rank, article in enumerate(summaries, start=1):
            title = article.get("title", f"Article {rank}")
            url = article.get("url", "")
            source = article.get("source", urlparse(url).netloc if url else "")
            published = article.get("published", "")
            summary = article.get("summary", "")

            pub_str = f" | **Published:** {published}" if published else ""
            lines += [
                f"## {rank}. {title}",
                f"**Source:** {source}{pub_str}  ",
                f"**URL:** {url}",
                "",
                summary,
                "",
                "---",
                "",
            ]

        newsletter_md = "\n".join(lines)

        # Write to output/news_<week>_<year>.md
        filename = f"news_{week_num:02d}_{year}.md"
        write_result = self.file_writer.write_file(filename, newsletter_md)

        if not write_result["success"]:
            raise RuntimeError(
                f"Failed to write newsletter: {write_result.get('error')}"
            )

        output_path = os.path.join(os.path.abspath(OUTPUT_DIR), filename)
        logger.info("Stage 7 complete: newsletter written to %s", output_path)

        self._cleanup_excluded_articles(summaries)

        return output_path

    def _cleanup_excluded_articles(self, included: List[Dict[str, Any]]) -> None:
        """
        Remove article files (and their _meta.json sidecars) from the articles
        week directory that were NOT included in the final newsletter.
        The evaluate_<week>.md file is always preserved.
        """
        articles_week_dir = getattr(self, "_articles_week_dir", None)
        if not articles_week_dir or not os.path.isdir(articles_week_dir):
            return

        week_num = getattr(self, "_week_num", None)

        # Build the set of absolute paths that must be kept
        keep: set = set()
        for a in included:
            lp = a.get("local_path", "")
            if lp:
                keep.add(os.path.abspath(lp))
                keep.add(os.path.abspath(lp.replace(".md", "_meta.json")))

        # Always keep the evaluation file
        if week_num is not None:
            keep.add(os.path.abspath(os.path.join(articles_week_dir, f"evaluate_{week_num}.md")))

        removed = 0
        for fname in os.listdir(articles_week_dir):
            fpath = os.path.abspath(os.path.join(articles_week_dir, fname))
            if fpath not in keep:
                try:
                    os.remove(fpath)
                    logger.debug("Removed excluded article file: %s", fname)
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", fpath, exc)

        logger.info("Cleanup: removed %d excluded article file(s) from %s", removed, articles_week_dir)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        Execute pipeline stages 1–6 (search → evaluate).
        Stops after writing the evaluation file.
        Returns the absolute path of the evaluation Markdown file.

        To generate the final newsletter run:
            python -m maki_newsletter.generate
        """
        logger.info("=" * 60)
        logger.info("Newsletter pipeline started")
        logger.info("=" * 60)

        candidates = self.stage_search()
        if not candidates:
            raise RuntimeError("Stage 1 returned no article candidates — check network connectivity.")

        trend_articles, trending_keywords = self.stage_trends()

        # Merge trend (Reddit) articles into candidates, deduplicating by URL
        seen_urls = {a["url"] for a in candidates}
        for a in trend_articles:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                candidates.append(a)
        logger.info("Combined candidate pool: %d articles", len(candidates))

        downloaded = self.stage_download(candidates)
        if not downloaded:
            raise RuntimeError("Stage 2 returned no downloaded articles — all fetches failed.")

        enriched = self.stage_read(downloaded)
        if not enriched:
            raise RuntimeError("Stage 3 returned no enriched articles.")

        top10 = self.stage_rank(enriched, trending_keywords=trending_keywords)
        summaries = self.stage_summarize(top10)
        eval_path = self.stage_evaluate(summaries, trending_keywords=trending_keywords)

        logger.info("=" * 60)
        logger.info("Evaluation complete → %s", eval_path)
        logger.info("Review the evaluation file, then run:")
        logger.info("  python -m maki_newsletter.generate")
        logger.info("=" * 60)
        return eval_path

    def run_generate(self) -> str:
        """
        Load persisted summaries JSON and write the final newsletter.
        Called by generate.py — not part of the normal pipeline run.

        All articles in the summaries JSON are included — the user is expected
        to have reviewed and edited the file manually before running this step.
        Returns the absolute path of the generated newsletter file.
        """
        from glob import glob

        output_abs = os.path.abspath(OUTPUT_DIR)
        pattern = os.path.join(output_abs, "summaries_*.json")
        candidates = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            raise RuntimeError(
                f"No summaries JSON found in {output_abs}. "
                "Run the pipeline first: python -m maki_newsletter.main"
            )

        json_path = candidates[0]
        logger.info("Loading summaries from %s", json_path)
        with open(json_path, encoding="utf-8") as fh:
            all_summaries = json.load(fh)

        if not all_summaries:
            raise RuntimeError(
                f"Summaries file {json_path} is empty. "
                "Re-run the pipeline or check the file."
            )

        # Restore week/year from filename (summaries_WW_YYYY.json)
        basename = os.path.basename(json_path)
        try:
            parts = basename.replace("summaries_", "").replace(".json", "").split("_")
            self._week_num = int(parts[0])
            self._year = int(parts[1])
        except (IndexError, ValueError):
            now = datetime.now(timezone.utc)
            self._week_num = now.isocalendar()[1]
            self._year = now.year

        logger.info("Generating newsletter from all %d articles in %s", len(all_summaries), json_path)
        return self.stage_write(all_summaries)
