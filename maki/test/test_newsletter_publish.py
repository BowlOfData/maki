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


if __name__ == "__main__":
    unittest.main()
