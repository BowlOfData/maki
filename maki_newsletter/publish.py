"""
publish.py — Publish a newsletter week to Altervista WordPress

Reads summaries_WW_YYYY.json, builds a single WordPress Page with each
article anchored by its title slug, publishes (or updates) it via the
WordPress REST API, then back-fills each article's `altervista_url` field
in the JSON so other tools can link directly to a specific article.

Each article section contains:
  - Title (linked to original source) as an anchor heading
  - Source badge and quality score
  - Main topic
  - TL;DR: the 2-sentence summary from the pipeline
  - Full article resume: extracted from the downloaded .md file
  - Technology tags
  - Prominent "Read full article →" link

The published page is placed as a child of the "Newsletter" parent page
(auto-discovered by slug, or pinned via ALTERVISTA_NEWSLETTER_PAGE_ID in .env).

Usage
-----
    python -m maki_newsletter.publish                        # latest week
    python -m maki_newsletter.publish --week 18 --year 2026
    python -m maki_newsletter.publish --week 18 --year 2026 --dry-run

Credentials are loaded from .env (same file used by the pipeline):
    ALTERVISTA_SITE_URL       https://yourblog.altervista.org
    ALTERVISTA_USERNAME       WordPress username
    ALTERVISTA_APP_PASSWORD   Application Password (spaces OK)

Optional:
    ALTERVISTA_NEWSLETTER_PAGE_ID   Parent page ID (skips auto-discovery)
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

from maki_newsletter.config import (
    ALTERVISTA_APP_PASSWORD,
    ALTERVISTA_NEWSLETTER_PAGE_ID,
    ALTERVISTA_SITE_URL,
    ALTERVISTA_USERNAME,
    ARTICLES_DIR,
    LLM_MODEL,
    MAX_ARTICLE_CHARS,
    OLLAMA_HOST,
    OUTPUT_DIR,
)
from maki.agents.agent_manager import AgentManager
from maki.makiLLama import MakiLLama

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

HERE        = Path(__file__).parent
OUTPUT_PATH = HERE / OUTPUT_DIR
ARTICLES_PATH = HERE / ARTICLES_DIR
API_BASE    = f"{ALTERVISTA_SITE_URL.rstrip('/')}/wp-json/wp/v2"

# Max paragraphs from the full article body to include in the resume
MAX_RESUME_PARAGRAPHS = 8
MAX_RESUME_WORDS = 500

_RESUME_AGENT: Optional[object] = None

# ---------------------------------------------------------------------------
# Filename helpers (mirrors pipeline._slug)
# ---------------------------------------------------------------------------

def _url_slug(url: str) -> str:
    """Derive the .md filename stem from a URL (matches pipeline._slug)."""
    parsed = urlparse(url)
    raw = (parsed.netloc + parsed.path).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return slug[:80] or "article"


def _find_article_md(url: str, week: int) -> Optional[Path]:
    """
    Locate the downloaded .md file for a given article URL by scanning
    all month subdirectories under ARTICLES_PATH for a directory named
    after the week number.
    """
    stem = _url_slug(url)
    if not ARTICLES_PATH.is_dir():
        return None
    for month_dir in sorted(ARTICLES_PATH.iterdir()):
        if not month_dir.is_dir():
            continue
        week_dir = month_dir / str(week)
        if not week_dir.is_dir():
            continue
        candidate = week_dir / f"{stem}.md"
        if candidate.exists():
            return candidate
    return None


def _extract_resume(md_path: Path, max_paragraphs: int = MAX_RESUME_PARAGRAPHS) -> str:
    """
    Extract a clean multi-paragraph resume from a downloaded Markdown article.

    Strips:
      - Markdown headings (#, ##, …)
      - Image tags (![…](…))
      - Inline links → keeps only link text
      - Bold/italic markers
      - Horizontal rules
      - Author/byline lines (short lines after the main body)

    Returns plain-text paragraphs joined by double newlines, capped at
    max_paragraphs non-empty paragraphs.
    """
    raw = md_path.read_text(encoding="utf-8", errors="replace")

    # Remove the first heading (title) — already shown as the <h2>
    raw = re.sub(r"^#[^\n]*\n", "", raw, count=1).strip()

    paragraphs: list[str] = []
    for block in re.split(r"\n{2,}", raw):
        block = block.strip()
        if not block:
            continue
        # Skip headings
        if re.match(r"^#{1,6}\s", block):
            continue
        # Skip horizontal rules
        if re.match(r"^[-*_]{3,}$", block):
            continue
        # Skip pure image lines
        if re.match(r"^!\[", block):
            continue

        # Strip markdown syntax
        block = re.sub(r"!\[.*?\]\(.*?\)", "", block)           # images
        block = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", block)  # links → text
        block = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", block)  # bold/italic
        block = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", block)    # underscore emphasis
        block = re.sub(r"`[^`]*`", "", block)                    # inline code
        block = re.sub(r"\s+", " ", block).strip()

        if len(block) < 40:  # skip very short byline / caption fragments
            continue

        paragraphs.append(block)
        if len(paragraphs) >= max_paragraphs:
            break

    return "\n\n".join(paragraphs)


def _clean_resume_text(text: str) -> str:
    """Normalize whitespace for publish-page article resumes."""
    return re.sub(r"[ \t]+", " ", (text or "").strip())


def _extract_resume_sentences(text: str) -> list[str]:
    """Return complete sentences, preserving ending punctuation."""
    cleaned = _clean_resume_text(text)
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?](?:['\")\]]+)?", cleaned)]


def _trim_resume_to_word_limit(text: str, max_words: int = MAX_RESUME_WORDS) -> str:
    """Trim a resume to *max_words* using whole sentences only."""
    sentences = _extract_resume_sentences(text)
    if not sentences:
        words = _clean_resume_text(text).split()
        return " ".join(words[:max_words]).strip()

    kept: list[str] = []
    count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if kept and count + len(sentence_words) > max_words:
            break
        if not kept and len(sentence_words) > max_words:
            return " ".join(sentence_words[:max_words]).strip()
        kept.append(sentence)
        count += len(sentence_words)
    return " ".join(kept).strip()


def _get_resume_agent():
    """Lazily create the agent used to summarize article bodies for publishing."""
    global _RESUME_AGENT
    if _RESUME_AGENT is None:
        llm = MakiLLama(model=LLM_MODEL, base_url=OLLAMA_HOST)
        manager = AgentManager(llm)
        manager.add_agent(
            name="publish_resume_agent",
            role="senior technical editor",
            instructions=(
                "You write polished, professional article summaries for a technical newsletter website. "
                f"Produce a consistent resume of no more than {MAX_RESUME_WORDS} words. "
                "Use a neutral editorial tone and only information present in the source article. "
                "Keep the structure consistent across articles: begin with the central announcement or claim, "
                "then cover the key technical details, evidence, constraints, or implications discussed in the article. "
                "Write in 2 to 4 short paragraphs, use complete sentences, and keep paragraph flow natural. "
                "Do not use bullet points, headings, marketing language, direct address, or filler phrases."
            ),
        )
        _RESUME_AGENT = manager.get_agent("publish_resume_agent")
    return _RESUME_AGENT


def _build_professional_resume(md_path: Path) -> str:
    """
    Generate a consistent long-form resume for the publish page.

    Falls back to cleaned article paragraphs when the LLM is unavailable or
    returns unusable output.
    """
    fallback = _extract_resume(md_path)
    try:
        raw = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raw = ""

    content = raw[:MAX_ARTICLE_CHARS].strip()
    if not content:
        return _trim_resume_to_word_limit(fallback)

    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem.replace("_", " ")
    task = (
        f"Write a professional and consistent article resume in no more than {MAX_RESUME_WORDS} words.\n\n"
        "Requirements:\n"
        "- Use only facts and themes present in the article.\n"
        "- Keep a neutral, publication-ready tone.\n"
        "- Use 2 to 4 short paragraphs.\n"
        "- Explain the main development first, then the relevant technical details and practical implications.\n"
        "- Do not use bullet points, headings, or generic filler.\n\n"
        f"Title: {title}\n\n"
        f"Article:\n{content}"
    )

    try:
        response = _clean_resume_text(_get_resume_agent().execute_task(task))
    except Exception:
        response = ""

    if not response:
        return _trim_resume_to_word_limit(fallback)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", response) if p.strip()]
    normalized = "\n\n".join(paragraphs) if paragraphs else response
    normalized = _trim_resume_to_word_limit(normalized)
    if not normalized:
        return _trim_resume_to_word_limit(fallback)
    return normalized


# ---------------------------------------------------------------------------
# Anchor / title slugify
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _summaries_path(week: int, year: int) -> Path:
    return OUTPUT_PATH / f"summaries_{week:02d}_{year}.json"


def _published_meta_path(week: int, year: int) -> Path:
    return OUTPUT_PATH / f"published_{week:02d}_{year}.json"


def _load_summaries(week: int, year: int) -> list[dict]:
    path = _summaries_path(week, year)
    if not path.exists():
        sys.exit(f"Summaries file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_summaries(week: int, year: int, data: list[dict]) -> None:
    _summaries_path(week, year).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_meta(week: int, year: int) -> dict:
    path = _published_meta_path(week, year)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_meta(week: int, year: int, meta: dict) -> None:
    _published_meta_path(week, year).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _find_latest_week() -> tuple[int, int]:
    files = sorted(OUTPUT_PATH.glob("summaries_*.json"))
    if not files:
        sys.exit(f"No summaries files found in {OUTPUT_PATH}")
    match = re.fullmatch(r"summaries_(\d{2})_(\d{4})\.json", files[-1].name)
    if not match:
        sys.exit(f"Cannot parse filename: {files[-1].name}")
    return int(match.group(1)), int(match.group(2))


# ---------------------------------------------------------------------------
# WordPress helpers
# ---------------------------------------------------------------------------

def _require_env() -> None:
    missing = [k for k, v in {
        "ALTERVISTA_SITE_URL":      ALTERVISTA_SITE_URL,
        "ALTERVISTA_USERNAME":      ALTERVISTA_USERNAME,
        "ALTERVISTA_APP_PASSWORD":  ALTERVISTA_APP_PASSWORD,
    }.items() if not v]
    if missing:
        sys.exit(
            f"Missing environment variables: {', '.join(missing)}\n"
            "Set them in .env (maki project root) before running."
        )


def _auth_header() -> str:
    token = b64encode(f"{ALTERVISTA_USERNAME}:{ALTERVISTA_APP_PASSWORD}".encode()).decode()
    return f"Basic {token}"


def _api_request(
    method: str,
    endpoint: str,
    payload: Optional[dict] = None,
    fatal: bool = True,
) -> Optional[dict]:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    body = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": _auth_header(),
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "User-Agent":    "BowlOfData-Publisher/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        if fatal:
            sys.exit(f"WordPress API error {exc.code} {method} {url}\n{body_text}")
        print(f"  Warning: API {exc.code} on {method} {endpoint} — {body_text[:120]}")
        return None
    except urllib.error.URLError as exc:
        if fatal:
            sys.exit(f"WordPress API connection error {method} {url}: {exc}")
        print(f"  Warning: connection error on {method} {endpoint} — {exc}")
        return None


def _get_or_create_newsletter_parent() -> int:
    if ALTERVISTA_NEWSLETTER_PAGE_ID:
        return int(ALTERVISTA_NEWSLETTER_PAGE_ID)

    pages = _api_request("GET", "pages?slug=newsletter&per_page=1")
    if pages:
        pid = pages[0]["id"]
        print(f"  Found Newsletter parent page (id={pid})")
        return pid

    print("  Creating 'Newsletter' parent page…")
    page = _api_request("POST", "pages", {
        "title":   "Newsletter",
        "slug":    "newsletter",
        "status":  "publish",
        "content": "<p>Weekly tech digest powered by Bowl of Data.</p>",
    })
    pid = page["id"]
    print(f"  Created Newsletter parent page (id={pid})")
    return pid


def _add_to_nav_menu(page_id: int, page_title: str, page_url: str) -> None:
    """
    Add the published page as a child menu item of 'Newsletter' in the
    primary navigation menu.  Uses the WP REST API menus endpoint (WP 5.9+).
    Failures are non-fatal: a warning is printed but publishing continues.
    """
    # 1. List all registered nav menus
    menus = _api_request("GET", "menus?per_page=100", fatal=False)
    if not menus:
        print("  Warning: menus REST endpoint unavailable — add the page to the "
              "Newsletter menu manually in WP Admin → Appearance → Menus")
        return

    # Prefer a menu whose slug hints at being the primary/main nav
    primary_slugs = {"primary", "main", "header-menu", "main-menu", "top-menu", "primary-menu"}
    menu = next(
        (m for m in menus if m.get("slug", "") in primary_slugs),
        menus[0],
    )
    menu_id = menu["id"]
    print(f"  Using nav menu '{menu.get('name', menu_id)}' (id={menu_id})")

    # 2. Fetch all items in that menu
    items = _api_request("GET", f"menu-items?menus={menu_id}&per_page=100", fatal=False)
    if items is None:
        print("  Warning: could not fetch menu items — skipping menu update")
        return

    # 3. Find the Newsletter menu item (match by title text)
    def _item_title(item: dict) -> str:
        t = item.get("title", "")
        return (t.get("rendered", "") if isinstance(t, dict) else t).lower()

    newsletter_item = next(
        (item for item in items if "newsletter" in _item_title(item)),
        None,
    )
    if not newsletter_item:
        print("  Warning: 'Newsletter' menu item not found — "
              "skipping menu update (add it manually if needed)")
        return
    newsletter_item_id = newsletter_item["id"]

    # 4. Skip if a menu item for this page already exists
    if any(item.get("object_id") == page_id for item in items):
        print(f"  Menu item for this page already exists — skipping")
        return

    # 5. Create the child menu item
    result = _api_request("POST", "menu-items", {
        "title":    page_title,
        "url":      page_url,
        "type":     "post_type",
        "object":   "page",
        "object_id": page_id,
        "menus":    menu_id,
        "parent":   newsletter_item_id,
        "status":   "publish",
    }, fatal=False)

    if result:
        print(f"  Added to nav menu under 'Newsletter' (menu_item_id={result['id']})")
    else:
        print("  Warning: failed to add menu item — add it manually in "
              "WP Admin → Appearance → Menus")


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------


def _paragraphs_to_html(text: str) -> str:
    """Wrap each paragraph of plain text in a <p> tag."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n".join(
        f'<p style="margin:0.7em 0;line-height:1.8;color:#2d2d2d;font-size:0.97em;">'
        f'{html_module.escape(p)}</p>'
        for p in paras
    )


# Palette tuned for a #d9d9d9 page background
_C = {
    "card_bg":      "#ffffff",
    "card_border":  "#c8c8c8",
    "card_shadow":  "0 2px 8px rgba(0,0,0,0.10), 0 1px 3px rgba(0,0,0,0.07)",
    "text":         "#1a1a1a",
    "text_muted":   "#606060",
    "accent":       "#c47f00",        # amber — readable on white and on #d9d9d9
    "accent_light": "#f5c518",        # yellow — used for decorative borders only
    "source_bg":    "#3d4a5c",
    "source_fg":    "#ffffff",
    "tldr_bg":      "#f9f6ee",
    "tldr_border":  "#c47f00",
    "tldr_label":   "#8a5a00",
    "tag_bg":       "#f2ede0",
    "tag_fg":       "#7a5100",
    "tag_border":   "#d4b06a",
    "btn_bg":       "#b87200",
    "btn_fg":       "#ffffff",
    "topic_border": "#c47f00",
    "topic_fg":     "#505050",
    "divider":      "#c4c4c4",
    "meta_fg":      "#555555",
}


def _build_page_html(articles: list[dict], week: int, year: int) -> str:
    c = _C
    lines: list[str] = [
        f'<p style="color:{c["meta_fg"]};font-size:0.88em;margin-bottom:1.75em;'
        f'letter-spacing:0.03em;">',
        f'  Week {week:02d} &middot; {year} &middot; {len(articles)} articles',
        '</p>',
        "",
    ]

    for i, a in enumerate(articles, 1):
        title   = (a.get("title") or "").strip()
        url     = (a.get("url") or "").strip()
        source  = (a.get("source") or "").strip()
        summary = (a.get("summary") or "").strip()
        topic   = (a.get("main_topic") or "").strip()
        tags    = [t.strip() for t in (a.get("technologies") or []) if t and t.strip()]
        anchor  = _slugify(title) if title else f"article-{i}"

        # --- Card open ----------------------------------------------------------
        lines += [
            f'<div id="{anchor}" style="background:{c["card_bg"]};'
            f'border:1px solid {c["card_border"]};border-radius:10px;'
            f'box-shadow:{c["card_shadow"]};margin-bottom:1.75em;overflow:hidden;">',
        ]

        # --- Card header (yellow top stripe + title) ----------------------------
        title_html = (
            f'<a href="{url}" target="_blank" rel="noopener"'
            f' style="color:{c["text"]};text-decoration:none;">'
            f'{html_module.escape(title)}</a>'
            if url else html_module.escape(title)
        )
        lines += [
            f'<div style="border-top:4px solid {c["accent_light"]};'
            f'padding:1.2em 1.5em 1em;">',
            f'<h2 style="margin:0 0 0.5em;font-size:1.25em;line-height:1.35;'
            f'color:{c["text"]};font-weight:700;">',
            f'  {title_html}',
            f'</h2>',
        ]

        # --- Source badge -------------------------------------------------------
        if source:
            lines.append(
                f'<p style="margin:0;">'
                f'<span style="background:{c["source_bg"]};color:{c["source_fg"]};'
                f'border-radius:4px;padding:2px 9px;font-size:0.78em;font-weight:600;">'
                f'{html_module.escape(source)}</span></p>'
            )

        lines.append('</div>')  # close card header

        # --- Card body ----------------------------------------------------------
        lines.append(
            f'<div style="padding:0 1.5em 0.5em;border-top:1px solid {c["card_border"]};">'
        )

        # Topic
        if topic:
            lines.append(
                f'<p style="border-left:3px solid {c["topic_border"]};'
                f'padding-left:0.7em;color:{c["topic_fg"]};font-style:italic;'
                f'margin:1em 0 0.9em;font-size:0.92em;line-height:1.55;">'
                f'{html_module.escape(topic)}</p>'
            )

        # TL;DR
        if summary:
            lines += [
                f'<div style="background:{c["tldr_bg"]};'
                f'border-left:3px solid {c["tldr_border"]};'
                f'border-radius:0 6px 6px 0;padding:0.75em 1em;margin:0.9em 0;">',
                f'<p style="margin:0 0 0.25em;font-size:0.72em;font-weight:700;'
                f'letter-spacing:0.07em;text-transform:uppercase;color:{c["tldr_label"]};">'
                f'TL;DR</p>',
                f'<p style="margin:0;line-height:1.7;color:{c["text"]};font-size:0.95em;">'
                f'{html_module.escape(summary)}</p>',
                '</div>',
            ]

        # Full article resume
        md_path = _find_article_md(url, week) if url else None
        if md_path:
            resume = _build_professional_resume(md_path)
            if resume:
                lines += [
                    '<div style="margin:0.9em 0 0.5em;">',
                    _paragraphs_to_html(resume),
                    '</div>',
                ]

        lines.append('</div>')  # close card body

        # --- Card footer (tags + button) ----------------------------------------
        lines.append(
            f'<div style="background:#efefef;border-top:1px solid {c["card_border"]};'
            f'padding:0.75em 1.5em;display:flex;flex-wrap:wrap;'
            f'align-items:center;gap:0.5em;">'
        )

        if tags:
            for t in tags:
                lines.append(
                    f'<span style="background:{c["tag_bg"]};color:{c["tag_fg"]};'
                    f'border:1px solid {c["tag_border"]};border-radius:99px;'
                    f'padding:2px 10px;font-size:0.73em;font-weight:600;">'
                    f'{html_module.escape(t)}</span>'
                )

        if url:
            lines += [
                f'<a href="{url}" target="_blank" rel="noopener"'
                f' style="margin-left:auto;display:inline-block;'
                f'background:{c["btn_bg"]};color:{c["btn_fg"]};'
                f'border-radius:6px;padding:0.4em 1.1em;font-weight:700;'
                f'font-size:0.82em;text-decoration:none;white-space:nowrap;">'
                f'Read full article &rarr;</a>',
            ]

        lines.append('</div>')  # close card footer
        lines.append('</div>')  # close card
        lines.append('')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main publish flow
# ---------------------------------------------------------------------------

def publish(week: int, year: int, dry_run: bool = False) -> None:
    _require_env()

    print(f"Loading summaries for Week {week:02d} · {year}…")
    articles = _load_summaries(week, year)
    print(f"  {len(articles)} articles")

    page_title   = f"Newsletter — Week {week:02d} · {year}"
    page_slug    = f"newsletter-week-{week:02d}-{year}"
    page_content = _build_page_html(articles, week, year)
    meta         = _load_meta(week, year)
    existing_id: Optional[int] = meta.get("page_id")

    if dry_run:
        print(f"\n[dry-run] Would {'update' if existing_id else 'create'} page:")
        print(f"  Title  : {page_title}")
        print(f"  Slug   : {page_slug}")
        print(f"  Length : {len(page_content)} chars")
        print("\n  Articles:")
        for i, a in enumerate(articles, 1):
            title  = (a.get("title") or "").strip()
            url    = (a.get("url") or "").strip()
            anchor = _slugify(title) if title else f"article-{i}"
            md     = _find_article_md(url, week) if url else None
            status = f"full resume ({md.name})" if md else "TL;DR only (no .md file)"
            print(f"    {i}. #{anchor}")
            print(f"       {title[:70]}")
            print(f"       content: {status}")
        print("\n[dry-run] No changes written.")
        return

    parent_id = _get_or_create_newsletter_parent()

    payload = {
        "title":   page_title,
        "slug":    page_slug,
        "content": page_content,
        "status":  "publish",
        "parent":  parent_id,
    }

    if existing_id:
        print(f"  Updating existing page (id={existing_id})…")
        page = _api_request("POST", f"pages/{existing_id}", payload)
    else:
        existing = _api_request("GET", f"pages?slug={page_slug}&per_page=1")
        if existing:
            existing_id = existing[0]["id"]
            print(f"  Found existing page by slug (id={existing_id}), updating…")
            page = _api_request("POST", f"pages/{existing_id}", payload)
        else:
            print("  Creating new page…")
            page = _api_request("POST", "pages", payload)

    page_id  = page["id"]
    page_url = page.get("link", "").rstrip("/")
    print(f"  Published: {page_url}  (id={page_id})")

    _save_meta(week, year, {"page_id": page_id, "page_url": page_url})

    print("  Updating navigation menu…")
    _add_to_nav_menu(page_id, page_title, page_url)

    for i, a in enumerate(articles):
        title  = (a.get("title") or "").strip()
        anchor = _slugify(title) if title else f"article-{i + 1}"
        articles[i]["altervista_url"] = f"{page_url}#{anchor}"

    _save_summaries(week, year, articles)
    print(f"  Back-filled altervista_url for {len(articles)} articles")
    print(f"\nDone. Page live at: {page_url}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish newsletter week to Altervista WordPress"
    )
    parser.add_argument("--week",    type=int, help="Week number (default: latest)")
    parser.add_argument("--year",    type=int, help="Year (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without publishing")
    args = parser.parse_args()

    if args.week and args.year:
        week, year = args.week, args.year
    elif args.week or args.year:
        parser.error("Provide both --week and --year, or neither.")
    else:
        week, year = _find_latest_week()
        print(f"Auto-selected: Week {week:02d} · {year}")

    publish(week, year, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
