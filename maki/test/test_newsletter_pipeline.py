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

    def test_stage_summarize_replaces_low_signal_cached_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            summary_path = article_path.replace(".md", "_summary.txt")

            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nKernel maintainers shipped a new isolation mechanism for containers.")
            with open(summary_path, "w", encoding="utf-8") as fh:
                fh.write("r/netsec · 20 upvotes")

            agent = SimpleNamespace(
                execute_task=lambda task: (
                    "Linux maintainers introduced a new container isolation mechanism. "
                    "It matters because it reduces the blast radius of kernel-level escapes."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(get_agent=lambda name: agent)

            summaries = pipeline.stage_summarize([{
                "title": "New Linux container isolation work",
                "url": "https://example.com/article",
                "snippet": "A new kernel isolation feature landed for container workloads.",
                "local_path": article_path,
            }])

            self.assertEqual(len(summaries), 1)
            self.assertNotIn("upvotes", summaries[0]["summary"].lower())
            self.assertIn("container isolation mechanism", summaries[0]["summary"].lower())

            with open(summary_path, encoding="utf-8") as fh:
                saved = fh.read()
            self.assertEqual(saved, summaries[0]["summary"])

    def test_stage_summarize_falls_back_when_model_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nDetailed markdown content for summarization.")

            agent = SimpleNamespace(execute_task=lambda task: "r/netsec · 20 upvotes")
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(get_agent=lambda name: agent)

            article = {
                "title": "Rust networking tool adds eBPF tracing",
                "url": "https://example.com/rust-ebpf",
                "snippet": "A Rust networking observability tool added eBPF tracing for live packet analysis.",
                "local_path": article_path,
            }
            summaries = pipeline.stage_summarize([article])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(
                summaries[0]["summary"],
                "A Rust networking observability tool added eBPF tracing for live packet analysis. "
                "Rust networking tool adds eBPF tracing."
            )
            self.assertNotIn("this matters because", summaries[0]["summary"].lower())

    def test_stage_summarize_discards_incomplete_third_fragment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nDetailed markdown content for summarization.")

            agent = SimpleNamespace(
                execute_task=lambda task: (
                    "A new TypeScript release improves inference for template-heavy codebases. "
                    "It matters because teams can remove custom workarounds and simplify internal tooling. "
                    "And this trailing fragment"
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(get_agent=lambda name: agent)

            article = {
                "title": "TypeScript improves inference",
                "url": "https://example.com/typescript",
                "snippet": "TypeScript improved inference in advanced generic patterns.",
                "local_path": article_path,
                "source": "TypeScript Blog",
            }
            summaries = pipeline.stage_summarize([article])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(
                summaries[0]["summary"],
                "A new TypeScript release improves inference for template-heavy codebases. "
                "It matters because teams can remove custom workarounds and simplify internal tooling."
            )

    def test_stage_summarize_uses_title_for_second_sentence_when_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nDetailed markdown content for summarization.")

            agent = SimpleNamespace(
                execute_task=lambda task: "Improved database replication landed for edge clusters"
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(get_agent=lambda name: agent)

            article = {
                "title": "Database vendor ships edge replication update",
                "url": "https://example.com/edge-db",
                "snippet": "Improved database replication landed for edge clusters.",
                "local_path": article_path,
            }
            summaries = pipeline.stage_summarize([article])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(
                summaries[0]["summary"],
                "Improved database replication landed for edge clusters. "
                "Database vendor ships edge replication update."
            )
            self.assertNotIn("this matters because", summaries[0]["summary"].lower())

    @staticmethod
    def _write_download_result(output_path: str, content: str):
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"success": True, "content": content}


if __name__ == "__main__":
    unittest.main()
