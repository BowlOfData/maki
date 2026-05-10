"""
Shared helpers for web-oriented plugins.
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse


def strip_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace for text extraction."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def now_utc() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def week_start_utc(now: Optional[datetime] = None) -> datetime:
    """Return Monday 00:00:00 UTC of the current ISO calendar week."""
    now = now or now_utc()
    return now - timedelta(
        days=now.weekday(),
        hours=now.hour,
        minutes=now.minute,
        seconds=now.second,
        microseconds=now.microsecond,
    )


def parse_published(date_str: str) -> Optional[datetime]:
    """Parse a published-date string into a timezone-aware datetime."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def struct_time_to_datetime(st) -> Optional[datetime]:
    """Convert a time.struct_time in UTC to a timezone-aware datetime."""
    if st is None:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
    except Exception:
        return None


MEDIA_EXTENSIONS = frozenset(
    ".jpg .jpeg .png .gif .webp .svg .bmp .tiff .tif "
    ".mp4 .mov .avi .webm .mkv .mp3 .wav .pdf".split()
)
MEDIA_HOSTS = frozenset([
    "i.redd.it", "preview.redd.it", "v.redd.it",
    "i.imgur.com", "imgur.com",
    "pbs.twimg.com", "video.twimg.com",
])


def is_media_url(url: str) -> bool:
    """Return True when *url* points to a direct media file rather than an article."""
    try:
        parsed = urlparse(url)
        if parsed.netloc in MEDIA_HOSTS:
            return True
        path_lower = parsed.path.lower()
        return any(path_lower.endswith(ext) for ext in MEDIA_EXTENSIONS)
    except Exception:
        return False


def is_current_week(date_str: str, now: Optional[datetime] = None) -> bool:
    """Return True if *date_str* falls within the current ISO calendar week."""
    dt = parse_published(date_str)
    if dt is None:
        return True
    now = now or now_utc()
    return week_start_utc(now) <= dt <= now
