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
                json.dump([{"url": "https://example.com/18", "short_summary": "older"}], fh)
            with open(newer_week, "w", encoding="utf-8") as fh:
                json.dump([{"url": "https://example.com/19", "short_summary": "newer"}], fh)

            os.utime(older_week, (2_000_000_000, 2_000_000_000))
            os.utime(newer_week, (1_000_000_000, 1_000_000_000))

            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.stage_write = lambda summaries: summaries[0]["url"]
            pipeline.stage_read = lambda articles: articles
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

    def test_stage_evaluate_backfills_missing_long_resume(self):
        pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
            newsletter_pipeline.NewsletterPipeline
        )
        pipeline._week_num = 18
        pipeline._year = 2026

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline._articles_week_dir = tmpdir
            article = {
                "title": "Example article",
                "url": "https://example.com/article",
                "short_summary": "A valid short summary that already exists for this article and stays comfortably within the expected range. It explains why the update matters to engineering teams evaluating the change.",
                "long_resume": "",
                "quality_score": 7,
            }
            refilled_article = {**article, "long_resume": "Paragraph one.\n\nParagraph two.\n\nParagraph three."}
            calls = {"count": 0}

            def _stage_read(items):
                calls["count"] += 1
                return [refilled_article]

            pipeline.stage_read = _stage_read
            pipeline._load_url_set = lambda path: set()
            pipeline._save_url_set = lambda path, urls: None

            with patch.object(newsletter_pipeline, "OUTPUT_DIR", tmpdir):
                eval_path = pipeline.stage_evaluate([article], trending_keywords=[])

            self.assertTrue(os.path.exists(eval_path))
            self.assertEqual(calls["count"], 1)
            with open(os.path.join(tmpdir, "summaries_18_2026.json"), encoding="utf-8") as fh:
                persisted = json.load(fh)
            self.assertEqual(persisted[0]["long_resume"], refilled_article["long_resume"])

    def test_run_generate_backfills_missing_long_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summaries_path = os.path.join(tmpdir, "summaries_19_2026.json")
            article = {
                "title": "Example article",
                "url": "https://example.com/article",
                "short_summary": "A valid short summary that already exists for this article and stays comfortably within the expected range. It explains why the update matters to engineering teams evaluating the change.",
                "long_resume": "",
            }
            with open(summaries_path, "w", encoding="utf-8") as fh:
                json.dump([article], fh)

            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            calls = {"count": 0}

            def _stage_read(items):
                calls["count"] += 1
                return [{**items[0], "long_resume": "Paragraph one.\n\nParagraph two.\n\nParagraph three."}]

            pipeline.stage_read = _stage_read
            pipeline.stage_write = lambda summaries: summaries[0]["long_resume"]
            pipeline._load_url_set = lambda path: set()
            pipeline._save_url_set = lambda path, urls: None
            pipeline._delete_article_files = lambda url: None

            with patch.object(newsletter_pipeline, "OUTPUT_DIR", tmpdir):
                result = pipeline.run_generate()

            self.assertEqual(calls["count"], 1)
            self.assertEqual(result, "Paragraph one.\n\nParagraph two.\n\nParagraph three.")

    def test_delete_article_files_removes_long_resume_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline._articles_week_dir = tmpdir
            base = os.path.join(tmpdir, "example_com_article")
            for suffix in (".md", "_meta.json", "_summary.txt", "_long_resume.txt"):
                with open(base + suffix, "w", encoding="utf-8") as fh:
                    fh.write("x")

            pipeline._delete_article_files("https://example.com/article")

            for suffix in (".md", "_meta.json", "_summary.txt", "_long_resume.txt"):
                self.assertFalse(os.path.exists(base + suffix))

    def test_stage_write_prefers_altervista_url_in_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: SimpleNamespace(
                    execute_task=lambda task: "Intro paragraph for the newsletter."
                )
            )
            pipeline.web_search = SimpleNamespace(fetch_pexels_image=lambda query, key: None)
            pipeline.file_writer = SimpleNamespace(
                write_file=lambda filename, content: self._capture_write(tmpdir, filename, content)
            )
            pipeline._cleanup_excluded_articles = lambda summaries: None
            pipeline._week_num = 18
            pipeline._year = 2026

            with patch.object(newsletter_pipeline, "OUTPUT_DIR", tmpdir):
                output_path = pipeline.stage_write([{
                    "title": "Example article",
                    "url": "https://example.com/source-article",
                    "altervista_url": "https://altervista.example/newsletter#example-article",
                    "source": "Example Source",
                    "short_summary": "Sentence one. Sentence two.",
                    "technologies": [],
                }])

            with open(output_path, encoding="utf-8") as fh:
                markdown = fh.read()

            self.assertIn(
                "[Example article](https://altervista.example/newsletter#example-article)",
                markdown,
            )
            self.assertNotIn("https://example.com/source-article", markdown)

    @staticmethod
    def _write_download_result(output_path: str, content: str):
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"success": True, "content": content}

    @staticmethod
    def _capture_write(tmpdir: str, filename: str, content: str):
        output_path = os.path.join(tmpdir, filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"success": True, "path": output_path}


if __name__ == "__main__":
    unittest.main()
