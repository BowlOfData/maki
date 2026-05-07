import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maki_newsletter import publish as newsletter_publish


class TestNewsletterPublish(unittest.TestCase):

    def test_build_page_html_includes_page_title(self):
        html = newsletter_publish._build_page_html([], 18, 2026)

        self.assertIn("Bowl of Data - Tech Newsletter &mdash; Week 18, 2026", html)

    def test_build_page_html_prefers_long_resume_from_summaries(self):
        article = {
            "title": "Edge database update",
            "url": "https://example.com/edge-db",
            "source": "Example Source",
            "summary": "Short summary sentence one. Short summary sentence two.",
            "long_resume": "Stored long resume from the pipeline. It contains the details publish.py should reuse.",
            "main_topic": "Databases",
            "technologies": ["Databases", "Replication"],
        }

        with patch.object(newsletter_publish, "_find_article_md", return_value=None):
            html = newsletter_publish._build_page_html([article], 18, 2026)

        self.assertIn("Stored long resume from the pipeline.", html)
        self.assertNotIn("TL;DR only", html)

    def test_build_page_html_falls_back_to_md_when_long_resume_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "article.md"
            md_path.write_text(
                "# Kernel release\n\n"
                "Kernel maintainers published a release candidate with scheduler changes.\n\n"
                "The article describes performance benchmarks and compatibility caveats for operators.",
                encoding="utf-8",
            )

            article = {
                "title": "Kernel release",
                "url": "https://example.com/kernel",
                "source": "Example Source",
                "summary": "Short summary sentence one. Short summary sentence two.",
                "long_resume": "",
                "main_topic": "Operating Systems",
                "technologies": ["Linux"],
            }

            with patch.object(newsletter_publish, "_find_article_md", return_value=md_path):
                html = newsletter_publish._build_page_html([article], 18, 2026)

            self.assertIn("scheduler changes", html)
            self.assertIn("performance benchmarks", html)


class TestModelReleasesSection(unittest.TestCase):

    _RELEASE = {
        "model_name": "GPT-5.5 Instant",
        "provider": "OpenAI",
        "release_date": "2026-05-05",
        "summary": "A new default model focused on smarter reasoning.",
        "key_features": ["Smarter reasoning", "Increased clarity"],
        "url": "https://openai.com/news/",
    }

    def _html(self, releases):
        with patch.object(newsletter_publish, "_find_article_md", return_value=None):
            return newsletter_publish._build_page_html([], 19, 2026, model_releases=releases)

    def test_model_releases_section_present(self):
        html = self._html([self._RELEASE])
        self.assertIn("AI Model Releases", html)

    def test_model_releases_count_in_header(self):
        html = self._html([self._RELEASE])
        self.assertIn("1 model release", html)

    def test_model_releases_plural_count(self):
        html = self._html([self._RELEASE, self._RELEASE])
        self.assertIn("2 model releases", html)

    def test_model_release_anchor_matches_slugified_name(self):
        html = self._html([self._RELEASE])
        slug = newsletter_publish._slugify("GPT-5.5 Instant")
        self.assertIn(f'id="{slug}"', html)

    def test_model_release_provider_badge(self):
        html = self._html([self._RELEASE])
        self.assertIn("OpenAI", html)

    def test_model_release_summary(self):
        html = self._html([self._RELEASE])
        self.assertIn("smarter reasoning", html)

    def test_model_release_key_features(self):
        html = self._html([self._RELEASE])
        self.assertIn("Smarter reasoning", html)
        self.assertIn("Increased clarity", html)

    def test_model_release_read_link(self):
        html = self._html([self._RELEASE])
        self.assertIn("https://openai.com/news/", html)
        self.assertIn("Read announcement", html)

    def test_no_model_releases_section_when_empty(self):
        html = self._html([])
        self.assertNotIn("AI Model Releases", html)

    def test_model_release_date_recent_hidden(self):
        release = dict(self._RELEASE, release_date="recent")
        html = self._html([release])
        self.assertNotIn("recent", html)


if __name__ == "__main__":
    unittest.main()
