"""
Newsletter pipeline — six dedicated stages, each implemented as an agent
or a dedicated function as required by the plan.

Stages
------
1. search_stage()       — dedicated function  — finds article candidates
2. download_stage()     — dedicated function  — downloads articles as Markdown
3. reader_agent         — LLM agent           — extracts metadata from each article
4. ranker_agent         — LLM agent           — ranks articles, selects top N
5. writer_agent         — LLM agent           — assembles the final newsletter

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
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from maki.agents.agent_manager import AgentManager
from maki.makiLLama import MakiLLama
from maki.plugins.file_reader.file_reader import FileReader
from maki.plugins.file_writer.file_writer import FileWriter
from maki.plugins.web_to_md.web_to_md import WebToMd

from .config import (
    MAX_RESUME_WORDS,
    ARTICLES_DIR,
    LLM_MODEL,
    MAX_ARTICLE_CHARS,
    MAX_CANDIDATES,
    MAX_HN_PER_QUERY,
    MAX_PER_FEED,
    MAX_REDDIT_PER_SUB,
    OLLAMA_HOST,
    OUTPUT_DIR,
    PEXELS_API_KEY,
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

def _repair_truncated_object(text: str) -> str:
    """
    Drop any trailing incomplete field from a truncated JSON object and close it.

    Walks the string tracking string context and brace depth. If the input is
    cut off before the closing ``}`` (e.g. the LLM hit a token limit mid-value),
    the function truncates at the last top-level comma and appends ``}`` so that
    all complete fields are preserved.  Returns an empty string when repair is
    not possible.
    """
    in_string = False
    escape_next = False
    depth = 0
    last_field_comma = -1

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[: i + 1]  # object closed normally — already valid
        elif ch == "," and depth == 1:
            last_field_comma = i

    if depth > 0 and last_field_comma > 0:
        return text[:last_field_comma] + "}"
    return ""


def _extract_json(text: str) -> Any:
    """
    Robustly extract the first JSON value (object or array) from an LLM response.
    Returns the parsed value, or None on failure.
    """
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = cleaned.find(start_char)
        if idx == -1:
            continue
        end_idx = cleaned.rfind(end_char)
        if end_idx == -1 or end_idx <= idx:
            continue
        try:
            return json.loads(cleaned[idx: end_idx + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: the response was truncated mid-value (e.g. token-limit cutoff).
    # Drop the incomplete last field and close the object so we keep everything else.
    idx = cleaned.find("{")
    if idx != -1:
        repaired = _repair_truncated_object(cleaned[idx:])
        if repaired:
            try:
                result = json.loads(repaired)
                logger.warning(
                    "_extract_json: response was truncated — recovered partial JSON "
                    "(dropped last incomplete field)"
                )
                return result
            except json.JSONDecodeError:
                pass

    logger.warning("_extract_json: could not parse JSON from LLM response")
    return None


def _truncate(content: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Return the first max_chars characters of a string."""
    return content[:max_chars] if len(content) > max_chars else content


def _iso_week_parts(dt: datetime) -> tuple[int, int]:
    """Return the ISO week number and ISO week-year for *dt*."""
    iso = dt.isocalendar()
    return iso[1], iso[0]


def _cap_sentences(text: str, max_sentences: int = 2) -> str:
    """Return *text* truncated to at most *max_sentences* sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])


def _extract_complete_sentences(text: str) -> List[str]:
    """Return only complete sentences from *text*, preserving punctuation."""
    cleaned = _clean_summary_text(text)
    if not cleaned:
        return []
    return re.findall(r"[^.!?]+[.!?](?:['\")\]]+)?", cleaned)


def _clean_summary_text(text: str) -> str:
    """Normalize whitespace and strip common formatting noise from summaries."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"^[>*#\-\s]+", "", cleaned)
    return cleaned.strip(" \"'")


def _is_low_signal_summary(text: str) -> bool:
    """
    Return True when *text* looks like feed metadata rather than a real summary.

    This catches cases such as ``r/netsec · 20 upvotes`` that occasionally leak
    in through snippets, caches, or model output.
    """
    cleaned = _clean_summary_text(text)
    if not cleaned:
        return True

    lower = cleaned.lower()
    if re.fullmatch(r"r/[a-z0-9_+-]+\s*[·|:-]\s*\d+\s+upvotes?", lower):
        return True

    words = re.findall(r"\b\w+\b", lower)
    if len(words) <= 8 and any(token in lower for token in ("upvotes", "points", "comments")):
        if "r/" in lower or "hacker news" in lower:
            return True

    return False


def _fallback_summary(article: Dict[str, Any]) -> str:
    """Return the best non-empty non-metadata fallback text for *article*."""
    snippet = _clean_summary_text(article.get("snippet", ""))
    if snippet and not _is_low_signal_summary(snippet):
        return snippet

    title = _clean_summary_text(article.get("title", ""))
    return title


def _title_as_sentence(article: Dict[str, Any]) -> str:
    """Convert the article title into a complete sentence."""
    title = _clean_summary_text(article.get("title", ""))
    if not title:
        return ""
    return title if re.search(r"[.!?]['\")\]]*$", title) else f"{title}."


def _article_derived_second_sentence(article: Dict[str, Any], first_sentence: str) -> str:
    """Build a second sentence using only article-derived text."""
    title_sentence = _title_as_sentence(article)
    if title_sentence and title_sentence.lower() != first_sentence.lower():
        return title_sentence

    snippet = _clean_summary_text(article.get("snippet", ""))
    snippet_sentences = [sentence.strip() for sentence in _extract_complete_sentences(snippet)]
    for sentence in snippet_sentences:
        if sentence.lower() != first_sentence.lower():
            return sentence

    if snippet:
        snippet_sentence = snippet if re.search(r"[.!?]['\")\]]*$", snippet) else f"{snippet.rstrip(',:;')}."
        if snippet_sentence.lower() != first_sentence.lower():
            return snippet_sentence

    return first_sentence


def _normalize_two_sentence_summary(text: str, article: Dict[str, Any]) -> str:
    """
    Return exactly two complete sentences without truncating mid-sentence.

    If fewer than two complete sentences are available, synthesize the second
    sentence from article-derived text rather than returning a fragment.
    """
    cleaned = _clean_summary_text(text)
    sentences = [sentence.strip() for sentence in _extract_complete_sentences(cleaned)]

    if not sentences and cleaned:
        cleaned = cleaned.rstrip(",:;")
        if cleaned:
            sentences = [f"{cleaned}."]

    if not sentences:
        title = _clean_summary_text(article.get("title", "")) or "This article highlights a technical development"
        sentences = [f"{title.rstrip(' .!?')}."]

    if len(sentences) == 1:
        sentences.append(_article_derived_second_sentence(article, sentences[0]))

    return " ".join(sentences[:2])


def _summaries_filename_parts(path: str) -> Optional[tuple[int, int]]:
    """Parse `summaries_WW_YYYY.json` and return `(week_num, iso_year)`."""
    match = re.fullmatch(r"summaries_(\d{2})_(\d{4})\.json", os.path.basename(path))
    if not match:
        return None
    week_num, iso_year = match.groups()
    return int(week_num), int(iso_year)


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
                "This includes cybersecurity news, vulnerability reports, threat intelligence, "
                "and articles about malware, exploits, or state-sponsored attacks — these are "
                "legitimate journalism topics for a technical newsletter. "
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

    def stage_search(
        self, trending_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch current-week articles from RSS feeds and HackerNews, guided by
        the trending keywords retrieved in stage_trends().

        RSS feeds are filtered so that only articles whose title or snippet
        mentions at least one domain seed keyword (IT, AI, security, blockchain)
        are kept.  HackerNews is searched using the specific trending queries
        when available, falling back to SEARCH_QUERIES otherwise.

        Returns up to MAX_CANDIDATES deduplicated article dicts.
        """
        logger.info("Stage 1: fetching trend-guided articles from RSS and HackerNews …")

        trending_keywords = trending_keywords or []

        # RSS filter: broad domain seeds ensure only IT/AI/security/blockchain
        # articles are fetched, discarding unrelated content from general feeds.
        rss_filter = list(TREND_SEED_KEYWORDS)

        # HN queries: use specific trending topics when available so that
        # HackerNews searches surface what is actually being discussed right now.
        # Fall back to the static SEARCH_QUERIES when trends are unavailable.
        hn_queries = trending_keywords[:len(SEARCH_QUERIES)] if trending_keywords else SEARCH_QUERIES

        seen_urls: set = set()
        candidates: List[Dict[str, Any]] = []
        downloaded_canonical_urls: set = set()  # tracks canonical URLs resolved during download

        def _add(articles: List[Dict]) -> None:
            for a in articles:
                url = (a.get("url") or "").rstrip("/")
                if url and url not in seen_urls and len(candidates) < MAX_CANDIDATES:
                    seen_urls.add(url)
                    candidates.append({**a, "url": url})

        # Primary: RSS feeds filtered to the domain areas
        _add(self.web_search.search_rss(
            RSS_FEEDS, max_per_feed=MAX_PER_FEED, keywords=rss_filter
        ))

        # Secondary: HackerNews searched with trending-derived queries
        for query in hn_queries:
            if len(candidates) >= MAX_CANDIDATES:
                break
            _add(self.web_search.search_hackernews(query, max_results=MAX_HN_PER_QUERY))

        logger.info(
            "Stage 1 complete: %d candidates found (rss_filter=%d keywords, hn_queries=%d)",
            len(candidates), len(rss_filter), len(hn_queries),
        )
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
        # and log each seed's results so they are visible in the run output.
        trending_keywords: list = []
        for seed, queries in google_trends.items():
            if queries:
                logger.info("  Google Trends [%s]: %s", seed, ", ".join(queries))
            else:
                logger.info("  Google Trends [%s]: (no results)", seed)
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
        week_num, iso_year = _iso_week_parts(now)   # e.g. 14, 2026
        articles_abs = os.path.abspath(
            os.path.join(ARTICLES_DIR, month_name, str(week_num))
        )
        os.makedirs(articles_abs, exist_ok=True)

        # Store for later stages
        self._articles_week_dir = articles_abs
        self._week_num = week_num
        self._year = iso_year
        downloaded_canonical_urls: set[str] = set()

        for article in candidates:
            url = article.get("url", "")
            if not url:
                continue

            filename = _slug(url) + ".md"
            output_path = os.path.join(articles_abs, filename)

            if os.path.exists(output_path):
                try:
                    existing_size = os.path.getsize(output_path)
                    if existing_size < 300:
                        logger.warning(
                            "Removing stale short article cache before re-download: %s (%d bytes)",
                            filename,
                            existing_size,
                        )
                        os.remove(output_path)
                    else:
                        logger.debug("Already exists, skipping download: %s", filename)
                        downloaded_canonical_urls.add(url.rstrip("/"))
                        downloaded.append(
                            {
                                **article,
                                "local_path": output_path,
                                "filename": filename,
                            }
                        )
                        continue
                except OSError as exc:
                    logger.warning(
                        "Could not validate cached article %s: %s — re-downloading",
                        filename,
                        exc,
                    )

            result = self.web_to_md.fetch_and_convert_to_md(url, output_path)

            if result["success"]:
                content_len = len(result.get("content", ""))
                if content_len < 300:
                    logger.debug("Skipping %s — content too short (%d chars)", url, content_len)
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except OSError as exc:
                        logger.warning("Could not remove short article cache %s: %s", output_path, exc)
                    continue

                # Use the canonical URL after redirects if it differs from the original
                canonical_url = (result.get("final_url") or url).rstrip("/")
                if canonical_url in downloaded_canonical_urls:
                    logger.debug("Skipping duplicate after redirect: %s → %s", url, canonical_url)
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    continue
                downloaded_canonical_urls.add(canonical_url)

                if canonical_url != url:
                    logger.info("Resolved canonical URL: %s → %s", url, canonical_url)

                downloaded.append(
                    {
                        **article,
                        "url":        canonical_url,
                        "local_path": output_path,
                        "filename":   filename,
                    }
                )
                logger.debug("Downloaded: %s → %s", canonical_url, filename)
            else:
                logger.warning("Failed to download %s: %s", url, result.get("error"))

            # Polite delay to reduce rate-limit pressure
            time.sleep(1.5)

        # Write a single manifest so run_from_stage3 can resume without re-downloading
        manifest_path = self._manifest_path()
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(downloaded, fh, indent=2, ensure_ascii=False)
            logger.info("Manifest written to %s", manifest_path)
        except OSError as exc:
            logger.warning("Could not write manifest %s: %s", manifest_path, exc)

        logger.info("Stage 2 complete: %d articles downloaded", len(downloaded))
        return downloaded

    # ------------------------------------------------------------------
    # Stage 3 — Read (LLM agent)
    # ------------------------------------------------------------------

    def _call_reader_agent(
        self,
        agent: Any,
        article: Dict[str, Any],
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call reader_agent with *content* and return the parsed metadata dict.
        Returns None when the response is empty, an exception is raised, or
        JSON cannot be extracted.
        """
        task = (
            "You are extracting metadata from a published news article for a tech newsletter. "
            "Return a JSON object with exactly these keys:\n"
            '  "main_topic": string (one sentence),\n'
            '  "key_points": array of up to 5 strings,\n'
            '  "technologies": array of technology/product names mentioned,\n'
            '  "quality_score": integer 0-10 (technical depth and relevance),\n'
            '  "short_summary": string (two sentences),\n'
            f'  "long_resume": string (max {MAX_RESUME_WORDS} words, complete sentences).\n\n'
            "Article content:\n"
            f"{content}"
        )
        raw_response = ""
        try:
            raw_response = agent.execute_task(task)
        except Exception as exc:
            logger.warning("reader_agent exception for %s: %s", article.get("url"), exc)
            return None

        if not raw_response.strip():
            logger.debug("reader_agent returned empty response for %s", article.get("url"))
            return None

        metadata = _extract_json(raw_response)
        if not isinstance(metadata, dict):
            logger.warning(
                "Could not parse JSON from LLM response for %s",
                article.get("url"),
            )
            logger.warning("LLM response was: %.500s", raw_response)
            return None
        return metadata

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
            meta_path = (os.path.splitext(local_path)[0] + "_meta.json") if local_path else ""

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
                with open(local_path, encoding="utf-8") as fh:
                    raw = fh.read()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", local_path, exc)
                continue

            metadata = self._call_reader_agent(agent, article, _truncate(raw))

            # Retry with half the content when the LLM returns empty or unparseable output
            if metadata is None:
                logger.info(
                    "Retrying %s with reduced content (%d chars) …",
                    article.get("url"), MAX_ARTICLE_CHARS // 2,
                )
                metadata = self._call_reader_agent(
                    agent, article, _truncate(raw, MAX_ARTICLE_CHARS // 2)
                )

            # Fall back to article metadata when LLM consistently fails rather
            # than dropping the article — it will land in the "Needs Review" bucket
            if metadata is None:
                logger.warning(
                    "reader_agent failed for %s after retry — using fallback metadata",
                    article.get("url"),
                )
                short = _normalize_two_sentence_summary(article.get("snippet", ""), article)
                metadata = {
                    "main_topic": article.get("title", ""),
                    "key_points": [],
                    "technologies": [],
                    "quality_score": 5,
                    "short_summary": short or _fallback_summary(article),
                    "long_resume": "",
                }

            # Discard low-quality articles immediately
            try:
                score = int(float(metadata.get("quality_score", 0)))
            except (TypeError, ValueError):
                score = 0
            if score < 4:
                logger.debug(
                    "Score %s < 4 for %s — removing article files",
                    metadata.get("quality_score"), article.get("url"),
                )
                self._delete_article_files(article.get("url", ""))
                continue

            # Only cache when both summary fields are present; truncated LLM
            # responses may have recovered partial JSON — leave the cache empty
            # so the next run retries the LLM for the missing fields.
            if meta_path and (metadata.get("short_summary") or "").strip() and (metadata.get("long_resume") or "").strip():
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
                seen_idx: set = set()
                for idx in indexes[: TOP_N]:
                    if 0 <= idx < len(enriched) and idx not in seen_idx:
                        seen_idx.add(idx)
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
    # Stage 5 — Evaluate (write evaluation file + stop)
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

        current_week_num, current_iso_year = _iso_week_parts(datetime.now(timezone.utc))
        week_num = getattr(self, "_week_num", current_week_num)
        year = getattr(self, "_year", current_iso_year)
        articles_week_dir = getattr(
            self, "_articles_week_dir",
            os.path.abspath(ARTICLES_DIR),
        )
        now = datetime.now(timezone.utc)

        # ---------- merge with existing week data ----------
        output_abs = os.path.abspath(OUTPUT_DIR)
        json_filename = f"summaries_{week_num:02d}_{year}.json"
        json_path = os.path.join(output_abs, json_filename)
        eval_filename = f"evaluate_{week_num}.md"
        eval_path = os.path.join(articles_week_dir, eval_filename)

        removed_path = self._removed_urls_path(week_num, year)
        checkpoint_path = self._checkpoint_path(week_num, year)

        # Load persistent removed set and last-write checkpoint
        removed_urls = self._load_url_set(removed_path)
        checkpoint_urls = self._load_url_set(checkpoint_path)

        existing: List[Dict[str, Any]] = []
        existing_loaded = False
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                existing_loaded = True
                logger.info("Loaded %d existing articles from %s", len(existing), json_path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not load existing summaries %s: %s — starting fresh", json_path, exc)

        # Detect articles removed by the user since the last pipeline write.
        # Only run when existing JSON was successfully loaded; otherwise a missing
        # summaries file would falsely mark all checkpoint URLs as removed.
        if existing_loaded and checkpoint_urls:
            existing_urls = {a.get("url", "") for a in existing}
            newly_removed = checkpoint_urls - existing_urls - removed_urls
            if newly_removed:
                logger.info("Detected %d manually removed article(s) — deleting files", len(newly_removed))
                for url in newly_removed:
                    self._delete_article_files(url)
                removed_urls |= newly_removed

        # Filter current-run summaries: skip any URL already marked as removed
        summaries = [a for a in summaries if a.get("url", "") not in removed_urls]

        if existing:
            # Also filter existing against removed_urls so a re-run can never
            # resurrect a removed article via the existing list
            existing = [a for a in existing if a.get("url", "") not in removed_urls]
            # Deduplicate existing in case the file already contains dupes
            seen_urls: set = set()
            deduped_existing: List[Dict[str, Any]] = []
            for a in existing:
                url = a.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduped_existing.append(a)
            existing = deduped_existing
            # Collect only articles whose URL has not been seen yet;
            # update seen_urls during iteration so duplicate entries in
            # summaries are also caught.
            new_articles: List[Dict[str, Any]] = []
            for a in summaries:
                url = a.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    new_articles.append(a)
            if new_articles:
                logger.info("Enriching evaluation with %d new article(s)", len(new_articles))
            else:
                logger.info("No new articles to add — evaluation already up to date")
            summaries = existing + new_articles

        # Re-summarise any article that is still missing a summary after the merge
        # (can happen when existing articles from a previous failed run are merged in)
        unsummarised = [
            a for a in summaries
            if not (a.get("short_summary") or "").strip() or not (a.get("long_resume") or "").strip()
        ]
        if unsummarised:
            logger.info("Re-summarising %d article(s) with missing short_summary or long_resume …", len(unsummarised))
            refilled = self.stage_read(unsummarised)
            refilled_by_url = {a.get("url", ""): a for a in refilled}
            unsummarised_urls = {a.get("url", "") for a in unsummarised}
            summaries = [
                refilled_by_url.get(a.get("url", ""), a)
                for a in summaries
                if a.get("url", "") not in unsummarised_urls or a.get("url", "") in refilled_by_url
            ]

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
            score = a.get("quality_score", "?")
            short_summary = a.get("short_summary", "")
            long_resume = a.get("long_resume", "")
            matched = _matched_trends(a)
            trend_str = (
                f"\n**Trending topics matched:** {', '.join(matched)}" if matched else ""
            )
            block = [
                f"### {rank}. {title}",
                f"**Source:** {source} | **Score:** {score}/10",
                f"**URL:** {url}{trend_str}",
                "",
                short_summary,
                "",
            ]
            if long_resume:
                block += [
                    "**Extended summary:**",
                    "",
                    long_resume,
                    "",
                ]
            block += ["---", ""]
            return block

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
        os.makedirs(articles_week_dir, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as fh:
            fh.write(eval_md)
        logger.info("Evaluation file written to %s", eval_path)

        # ---------- persist summaries JSON for generate.py ----------
        os.makedirs(output_abs, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summaries, fh, indent=2, ensure_ascii=False)
        logger.info("Summaries JSON written to %s", json_path)

        # ---------- update checkpoint and removed tracking files ----------
        self._save_url_set(checkpoint_path, {a.get("url", "") for a in summaries})
        self._save_url_set(removed_path, removed_urls)

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
        current_week_num, current_iso_year = _iso_week_parts(now)
        week_num = getattr(self, "_week_num", current_week_num)
        year = getattr(self, "_year", current_iso_year)

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

        # Derive a Pexels search query from the most common technologies
        tech_counter: Counter = Counter()
        for a in summaries:
            for t in a.get("technologies", []):
                if t:
                    tech_counter[t.lower()] += 1
        pexels_query = " ".join(t for t, _ in tech_counter.most_common(5)) or "technology innovation"
        cover_image_url = self.web_search.fetch_pexels_image(pexels_query, PEXELS_API_KEY)

        # Assemble Markdown
        intro_block = [
            "## Introduction",
            "",
            introduction,
            "",
        ]
        if cover_image_url:
            intro_block += [f"![Weekly Tech Highlight]({cover_image_url})", ""]
        intro_block += ["---", ""]

        lines: List[str] = [
            f"# Bowl of Data - Tech Newsletter — Week {week_num}, {year}",
            f"*Generated on {now.strftime('%Y-%m-%d')}*",
            "",
            "---",
            "",
            *intro_block,
        ]

        for rank, article in enumerate(summaries, start=1):
            title = article.get("title", f"Article {rank}")
            url = article.get("url", "")
            link_url = article.get("altervista_url") or url
            source = article.get("source", urlparse(url).netloc if url else "")
            summary = article.get("short_summary", "")

            lines += [
                f"## {rank}. [{title}]({link_url})",
                f"**Source:** {source}",
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

    def _manifest_path(self) -> str:
        """Return the absolute path of the download manifest for the current week."""
        week_num = getattr(self, "_week_num", 0)
        year = getattr(self, "_year", 0)
        articles_week_dir = getattr(self, "_articles_week_dir", os.path.abspath(ARTICLES_DIR))
        return os.path.join(articles_week_dir, f"manifest_{week_num:02d}_{year}.json")

    def _removed_urls_path(self, week_num: int, year: int) -> str:
        return os.path.join(os.path.abspath(OUTPUT_DIR), f"removed_{week_num:02d}_{year}.json")

    def _checkpoint_path(self, week_num: int, year: int) -> str:
        return os.path.join(os.path.abspath(OUTPUT_DIR), f"checkpoint_{week_num:02d}_{year}.json")

    def _load_url_set(self, path: str) -> set:
        """Load a JSON list of URLs from *path* into a set; return empty set on any error."""
        if not os.path.exists(path):
            return set()
        try:
            with open(path, encoding="utf-8") as fh:
                return set(json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load URL set %s: %s", path, exc)
            return set()

    def _save_url_set(self, path: str, urls: set) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(sorted(urls), fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Could not save URL set %s: %s", path, exc)

    def _delete_article_files(self, url: str) -> None:
        """Delete the article file and all sidecars for *url* from the current week dir."""
        articles_week_dir = getattr(self, "_articles_week_dir", None)
        if not articles_week_dir:
            return
        base = os.path.join(articles_week_dir, _slug(url))
        for fpath in (base + ".md", base + "_meta.json", base + "_summary.txt", base + "_long_resume.txt"):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug("Deleted removed article file: %s", fpath)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", fpath, exc)

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
                keep.add(os.path.abspath(os.path.splitext(lp)[0] + "_meta.json"))

        # Always keep the evaluation file and the manifest
        if week_num is not None:
            keep.add(os.path.abspath(os.path.join(articles_week_dir, f"evaluate_{week_num}.md")))
        keep.add(os.path.abspath(self._manifest_path()))

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

        # Trends run FIRST so their keywords guide what gets searched and downloaded
        trend_articles, trending_keywords = self.stage_trends()

        # Search using trends: RSS filtered to domain areas, HN queried with
        # specific trending topics so only relevant articles are discovered
        candidates = self.stage_search(trending_keywords)

        # Merge Reddit hot posts (already domain-filtered by subreddit choice)
        seen_urls = {a["url"] for a in candidates}
        for a in trend_articles:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                candidates.append(a)

        # Filter out URLs manually removed this week so they are not re-downloaded
        now = datetime.now(timezone.utc)
        current_week, current_year = _iso_week_parts(now)
        removed_urls = self._load_url_set(self._removed_urls_path(current_week, current_year))
        if removed_urls:
            before = len(candidates)
            candidates = [a for a in candidates if a["url"] not in removed_urls]
            logger.info("Filtered %d removed article(s) from candidate pool", before - len(candidates))

        logger.info("Combined candidate pool: %d articles", len(candidates))

        if not candidates:
            raise RuntimeError("No article candidates found — check network connectivity and trend fetch.")

        downloaded = self.stage_download(candidates)
        if not downloaded:
            raise RuntimeError("Stage 2: all article downloads failed — check network connectivity.")

        enriched = self.stage_read(downloaded)
        if not enriched:
            raise RuntimeError("Stage 3 returned no enriched articles.")

        top10 = self.stage_rank(enriched, trending_keywords=trending_keywords)
        eval_path = self.stage_evaluate(top10, trending_keywords=trending_keywords)

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
        candidates = glob(pattern)
        if not candidates:
            raise RuntimeError(
                f"No summaries JSON found in {output_abs}. "
                "Run the pipeline first: python -m maki_newsletter.main"
            )

        dated_candidates = [
            (parts, path)
            for path in candidates
            for parts in [_summaries_filename_parts(path)]
            if parts is not None
        ]
        if dated_candidates:
            dated_candidates.sort(key=lambda item: item[0], reverse=True)
            json_path = dated_candidates[0][1]
        else:
            logger.warning(
                "Could not parse summaries filenames by week/year; falling back to modification time"
            )
            json_path = max(candidates, key=os.path.getmtime)

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
            self._week_num, self._year = _iso_week_parts(now)

        # Restore _articles_week_dir from the first article that has a valid local_path
        if not getattr(self, "_articles_week_dir", None):
            for a in all_summaries:
                lp = a.get("local_path", "")
                if lp:
                    candidate = os.path.dirname(os.path.abspath(lp))
                    if os.path.isdir(candidate):
                        self._articles_week_dir = candidate
                        break

        # Detect articles removed from the JSON since the last pipeline write,
        # delete their files and update the tracking files
        week_num = self._week_num
        year = self._year
        removed_path = self._removed_urls_path(week_num, year)
        checkpoint_path = self._checkpoint_path(week_num, year)
        removed_urls = self._load_url_set(removed_path)
        checkpoint_urls = self._load_url_set(checkpoint_path)
        current_urls = {a["url"] for a in all_summaries if a.get("url")}

        if checkpoint_urls:
            newly_removed = checkpoint_urls - current_urls - removed_urls
            if newly_removed:
                logger.info(
                    "Detected %d article(s) removed from summaries JSON — deleting files",
                    len(newly_removed),
                )
                for url in newly_removed:
                    self._delete_article_files(url)
                removed_urls |= newly_removed
                self._save_url_set(removed_path, removed_urls)

        # Fill in missing or empty summaries
        missing = [
            a for a in all_summaries
            if not (a.get("short_summary") or "").strip() or not (a.get("long_resume") or "").strip()
        ]
        if missing:
            logger.info("Re-summarising %d article(s) with missing short_summary or long_resume …", len(missing))
            filled = self.stage_read(missing)
            filled_by_url = {a.get("url", ""): a for a in filled}
            all_summaries = [filled_by_url.get(a.get("url", ""), a) for a in all_summaries]

        # Persist the final state (pruned + re-summarised) so subsequent runs are consistent
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, indent=2, ensure_ascii=False)

        # Update checkpoint to reflect the current article set
        self._save_url_set(checkpoint_path, current_urls)

        logger.info("Generating newsletter from all %d articles in %s", len(all_summaries), json_path)
        return self.stage_write(all_summaries)

    def run_from_stage3(self, refetch_trends: bool = False) -> str:
        """
        Resume the pipeline from stage 3 (read → rank → evaluate) using the
        manifest written by stage_download at the end of the previous run.

        Targets the current ISO week — articles must already be present in the
        articles directory for that week.

        Returns the absolute path of the evaluation Markdown file.
        """
        now = datetime.now(timezone.utc)
        month_name = now.strftime("%B").lower()
        week_num, iso_year = _iso_week_parts(now)
        articles_abs = os.path.abspath(
            os.path.join(ARTICLES_DIR, month_name, str(week_num))
        )
        self._articles_week_dir = articles_abs
        self._week_num = week_num
        self._year = iso_year

        manifest_path = self._manifest_path()
        if not os.path.exists(manifest_path):
            raise RuntimeError(
                f"No manifest found at {manifest_path}. "
                "Run the full pipeline first: python -m maki_newsletter.main"
            )

        try:
            with open(manifest_path, encoding="utf-8") as fh:
                downloaded = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read manifest {manifest_path}: {exc}") from exc

        # Filter to articles whose local file still exists on disk
        valid = [a for a in downloaded if os.path.exists(a.get("local_path", ""))]
        if not valid:
            raise RuntimeError(
                f"No article files found for week {week_num}/{iso_year}. "
                "Re-run the full pipeline."
            )
        if len(valid) < len(downloaded):
            logger.warning(
                "Manifest lists %d articles but only %d files exist on disk — proceeding",
                len(downloaded), len(valid),
            )

        trending_keywords: List[str] = []
        if refetch_trends:
            _, trending_keywords = self.stage_trends()

        logger.info("Resuming from stage 3 with %d articles …", len(valid))
        enriched = self.stage_read(valid)
        if not enriched:
            raise RuntimeError("Stage 3 returned no enriched articles.")

        top10 = self.stage_rank(enriched, trending_keywords=trending_keywords)
        return self.stage_evaluate(top10, trending_keywords=trending_keywords)
