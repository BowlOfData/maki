"""
Web to Markdown Plugin for Maki Framework

Fetches a web page, extracts only the article title and body using Mozilla's
Readability algorithm (readability-lxml), then converts the result to clean
Markdown (html2text).  Navigation bars, footers, sidebars, ads, and other
page chrome are discarded before conversion.

Both readability-lxml and html2text are already listed in requirements.txt.
A regex-based fallback is used if either library is unavailable.
"""

import logging
import re
import time
import requests
from requests.adapters import HTTPAdapter
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from maki.plugins.file_writer.file_writer import FileWriter

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_MAX_RETRIES = 3
_RETRY_DELAYS = (2.0, 5.0, 10.0)  # seconds between attempts
_RETRYABLE_ERRORS = (
    "RemoteDisconnected",
    "ConnectionAborted",
    "ConnectionReset",
    "ConnectionError",
    "ChunkedEncodingError",
)

# Optional dependencies — imported once at module load so the cost is paid
# only when the module is first imported, not on every call.
try:
    from readability import Document as _ReadabilityDocument
    _READABILITY_AVAILABLE = True
except ImportError:
    _READABILITY_AVAILABLE = False

try:
    import html2text as _html2text_mod
    _HTML2TEXT_AVAILABLE = True
except ImportError:
    _HTML2TEXT_AVAILABLE = False


class WebToMd:
    """
    Fetches a web page and converts only its article content to Markdown.

    Uses readability-lxml to isolate the article body (discarding nav, footer,
    sidebars, ads) and html2text for clean HTML→Markdown conversion.
    """

    def __init__(self, maki_instance=None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "WebToMd plugin initialized (readability=%s, html2text=%s)",
            _READABILITY_AVAILABLE, _HTML2TEXT_AVAILABLE,
        )
        self.file_writer = FileWriter(maki_instance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_convert_to_md(self, url: str, output_file: str = None) -> Dict[str, Any]:
        """
        Fetch a web page and convert its article body to Markdown.

        Args:
            url:         The URL to fetch.
            output_file: Path to write the Markdown file.  Auto-generated from
                         the URL when omitted.

        Returns:
            Dict with keys: success, url, output_file, content, error, status_code.
        """
        if not isinstance(url, str) or not url.strip():
            return self._err(url, output_file, "URL must be a non-empty string")

        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return self._err(url, output_file, "Invalid URL format")
        except Exception as exc:
            return self._err(url, output_file, f"Invalid URL format: {exc}")

        result: Dict[str, Any] = {
            "success": False, "url": url, "output_file": output_file,
            "content": "", "error": None, "status_code": None,
        }

        try:
            response = self._fetch_with_retry(url)
            result["status_code"] = response.status_code

            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: Failed to fetch URL"
                return result

            content_type = response.headers.get("Content-Type", "")
            if not any(t in content_type for t in ("text/html", "application/xhtml", "text/xml", "application/xml")):
                result["error"] = f"Unsupported content type: {content_type}"
                return result

            title, article_html = self._extract_article(response.text, url)
            markdown_content = self._to_markdown(title, article_html)

            if output_file is None:
                path = parsed_url.path
                slug = re.sub(r"[^a-zA-Z0-9\-_.]", "_", path.split("/")[-1]) or "page"
                output_file = f"{parsed_url.netloc}_{slug}.md"
                result["output_file"] = output_file

            write_result = self.file_writer.write_file(output_file, markdown_content)
            if write_result["success"]:
                result["success"] = True
                result["content"] = markdown_content
                self.logger.info("Saved article to %s", output_file)
            else:
                result["error"] = f"Failed to write file: {write_result['error']}"

        except requests.exceptions.RequestException as exc:
            result["error"] = f"Request failed: {exc}"
            self.logger.error("Request failed for %s: %s", url, exc, exc_info=True)
        except Exception as exc:
            result["error"] = f"Unexpected error: {exc}"
            self.logger.error("Unexpected error for %s: %s", url, exc, exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _err(url: Any, output_file: Optional[str], msg: str) -> Dict[str, Any]:
        return {
            "success": False, "url": url, "output_file": output_file,
            "content": "", "error": msg, "status_code": None,
        }

    def _fetch_with_retry(self, url: str) -> requests.Response:
        """
        Fetch *url* with browser-like headers, retrying on transient connection
        errors (RemoteDisconnected, ConnectionAborted, etc.).

        Raises the last exception if all attempts are exhausted.
        """
        session = requests.Session()
        session.headers.update(_BROWSER_HEADERS)
        # Add a Referer that looks organic for news sites
        parsed = urlparse(url)
        session.headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
            try:
                response = session.get(url, timeout=25, allow_redirects=True)
                return response
            except Exception as exc:
                last_exc = exc
                exc_name = type(exc).__name__
                cause_name = type(exc.__cause__).__name__ if exc.__cause__ else ""
                is_retryable = any(
                    r in exc_name or r in cause_name or r in str(exc)
                    for r in _RETRYABLE_ERRORS
                )
                if not is_retryable or delay is None:
                    raise
                self.logger.warning(
                    "fetch attempt %d/%d failed for %s (%s) — retrying in %.0fs",
                    attempt, _MAX_RETRIES, url, exc, delay,
                )
                time.sleep(delay)

        raise last_exc

    def _extract_article(self, html: str, url: str = "") -> Tuple[str, str]:
        """
        Use readability-lxml to isolate the article title and body HTML,
        stripping navigation, headers, footers, sidebars, and ads.

        Returns (title, article_html).  Falls back to the full HTML when
        readability is unavailable or extraction fails.
        """
        if not _READABILITY_AVAILABLE:
            self.logger.debug(
                "readability-lxml not installed — skipping article extraction; "
                "install it with: pip install readability-lxml"
            )
            return "", html

        try:
            doc = _ReadabilityDocument(html, url=url)
            title = doc.title() or ""
            # doc.summary() returns a minimal HTML document containing only
            # the article body — all chrome has been removed.
            article_html = doc.summary(html_partial=False)
            return title.strip(), article_html
        except Exception as exc:
            self.logger.debug("readability extraction failed (%s) — using full HTML", exc)
            return "", html

    def _to_markdown(self, title: str, article_html: str) -> str:
        """
        Convert article HTML to Markdown.

        Uses html2text when available (clean, link-preserving output).
        Falls back to a regex-based converter otherwise.

        The article title is prepended as an H1 heading when present.
        """
        if _HTML2TEXT_AVAILABLE:
            converter = _html2text_mod.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True   # images are not useful in text newsletters
            converter.ignore_tables = False
            converter.body_width = 0         # disable hard line-wrapping
            converter.single_line_break = False
            body = converter.handle(article_html).strip()
        else:
            body = self._regex_to_markdown(article_html)

        if title:
            # Avoid duplicating the title when it already appears as the first
            # heading in the converted body.
            first_line = body.lstrip().split("\n", 1)[0].lstrip("#").strip()
            if first_line.lower() != title.lower():
                body = f"# {title}\n\n{body}"

        # Collapse runs of 3+ blank lines produced by some converters
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body.strip()

    @staticmethod
    def _regex_to_markdown(html: str) -> str:
        """Minimal regex HTML→Markdown fallback (used only when html2text is absent)."""
        h = html
        h = re.sub(r"<script.*?</script>", "", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<style.*?</style>",   "", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h1[^>]*>(.*?)</h1>", r"# \1\n",      h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h2[^>]*>(.*?)</h2>", r"## \1\n",     h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h3[^>]*>(.*?)</h3>", r"### \1\n",    h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h4[^>]*>(.*?)</h4>", r"#### \1\n",   h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h5[^>]*>(.*?)</h5>", r"##### \1\n",  h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<h6[^>]*>(.*?)</h6>", r"###### \1\n", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<p[^>]*>(.*?)</p>",   r"\1\n\n",      h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(
            r"<a\s+href=[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", r"[\2](\1)", h,
            flags=re.DOTALL | re.IGNORECASE,
        )
        h = re.sub(r"<strong>(.*?)</strong>", r"**\1**", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<b[^>]*>(.*?)</b>",      r"**\1**", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<em>(.*?)</em>",          r"*\1*",   h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<i[^>]*>(.*?)</i>",       r"*\1*",   h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<li[^>]*>(.*?)</li>",     r"- \1\n", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<(?:ul|ol)[^>]*>|</(?:ul|ol)>", "", h, flags=re.IGNORECASE)
        h = re.sub(r"<pre[^>]*>(.*?)</pre>",   r"```\n\1\n```\n", h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`",   h, flags=re.DOTALL | re.IGNORECASE)
        h = re.sub(r"<br\s*/?>", "\n", h, flags=re.IGNORECASE)
        h = re.sub(r"<[^>]+>", "", h)
        h = re.sub(r"\n{3,}", "\n\n", h)
        return h.strip()


def register_plugin(maki_instance=None):
    return WebToMd(maki_instance)