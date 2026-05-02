import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from maki_newsletter import pipeline as newsletter_pipeline


class TestNewsletterPipeline(unittest.TestCase):

    def test_iso_week_parts_uses_iso_year(self):
        week_num, iso_year = newsletter_pipeline._iso_week_parts(
            datetime(2021, 1, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(week_num, 53)
        self.assertEqual(iso_year, 2020)

    def test_summaries_filename_parts_parses_week_and_year(self):
        self.assertEqual(
            newsletter_pipeline._summaries_filename_parts("summaries_19_2026.json"),
            (19, 2026),
        )
        self.assertIsNone(
            newsletter_pipeline._summaries_filename_parts("summaries_latest.json")
        )

    def test_run_generate_prefers_latest_week_over_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            older_week = os.path.join(tmpdir, "summaries_18_2026.json")
            newer_week = os.path.join(tmpdir, "summaries_19_2026.json")

            with open(older_week, "w", encoding="utf-8") as fh:
                json.dump([{"url": "https://example.com/18", "summary": "older"}], fh)
            with open(newer_week, "w", encoding="utf-8") as fh:
                json.dump([{"url": "https://example.com/19", "summary": "newer"}], fh)

            os.utime(older_week, (2_000_000_000, 2_000_000_000))
            os.utime(newer_week, (1_000_000_000, 1_000_000_000))

            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.stage_write = lambda summaries: summaries[0]["url"]
            pipeline.stage_summarize = lambda missing: missing
            pipeline._load_url_set = lambda path: set()
            pipeline._save_url_set = lambda path, urls: None
            pipeline._delete_article_files = lambda url: None

            with patch.object(newsletter_pipeline, "OUTPUT_DIR", tmpdir):
                selected_url = pipeline.run_generate()

            self.assertEqual(selected_url, "https://example.com/19")

    def test_stage_download_replaces_stale_short_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.web_to_md = SimpleNamespace(
                fetch_and_convert_to_md=lambda url, output_path: self._write_download_result(
                    output_path, "# Long article\n\n" + ("content " * 80)
                )
            )

            article = {"url": "https://example.com/article", "title": "Example"}
            stale_path = os.path.join(tmpdir, "example_com_article.md")
            with open(stale_path, "w", encoding="utf-8") as fh:
                fh.write("too short")

            fixed_now = datetime(2026, 5, 2, tzinfo=timezone.utc)
            with patch.object(newsletter_pipeline, "ARTICLES_DIR", tmpdir), \
                 patch.object(newsletter_pipeline, "datetime") as mock_datetime, \
                 patch.object(newsletter_pipeline.time, "sleep", return_value=None):
                mock_datetime.now.return_value = fixed_now
                downloaded = pipeline.stage_download([article])

            self.assertEqual(len(downloaded), 1)
            self.assertGreater(os.path.getsize(downloaded[0]["local_path"]), 300)

    @staticmethod
    def _write_download_result(output_path: str, content: str):
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"success": True, "content": content}


if __name__ == "__main__":
    unittest.main()
