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
            long_resume_path = article_path.replace(".md", "_long_resume.txt")

            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\nKernel maintainers shipped a new isolation mechanism for containers.\n\n"
                    "The article explains how the design limits the impact of kernel-level escapes."
                )
            with open(summary_path, "w", encoding="utf-8") as fh:
                fh.write("r/netsec · 20 upvotes")

            short_calls = {"count": 0}

            def _short_execute(task: str) -> str:
                short_calls["count"] += 1
                if short_calls["count"] == 1:
                    return (
                        "Linux maintainers introduced a new container isolation mechanism. "
                        "It matters because it reduces the blast radius of kernel-level escapes."
                    )
                return (
                    "Linux maintainers introduced a new container isolation mechanism for container workloads and framed it as a practical hardening step for production systems. "
                    "It matters because operators can reduce the blast radius of kernel-level escapes while planning rollout changes around concrete isolation boundaries."
                )

            short_agent = SimpleNamespace(execute_task=_short_execute)
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "Linux maintainers introduced a new container isolation mechanism for container workloads. "
                    "The article describes how the approach reduces the blast radius of kernel-level escapes and "
                    "what changes operators need to understand before rollout."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

            summaries = pipeline.stage_summarize([{
                "title": "New Linux container isolation work",
                "url": "https://example.com/article",
                "snippet": "A new kernel isolation feature landed for container workloads.",
                "local_path": article_path,
            }])

            self.assertEqual(len(summaries), 1)
            self.assertNotIn("upvotes", summaries[0]["summary"].lower())
            self.assertIn("container isolation mechanism", summaries[0]["summary"].lower())
            self.assertIn("kernel-level escapes", summaries[0]["long_resume"].lower())
            self.assertEqual(short_calls["count"], 2)

            with open(summary_path, encoding="utf-8") as fh:
                saved = fh.read()
            self.assertEqual(saved, summaries[0]["summary"])
            with open(long_resume_path, encoding="utf-8") as fh:
                saved_resume = fh.read()
            self.assertEqual(saved_resume, summaries[0]["long_resume"])

    def test_stage_summarize_falls_back_when_model_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\nDetailed markdown content for summarization.\n\n"
                    "The release adds eBPF tracing for live packet analysis in production."
                )

            short_agent = SimpleNamespace(execute_task=lambda task: "r/netsec · 20 upvotes")
            long_agent = SimpleNamespace(execute_task=lambda task: "r/netsec · 20 upvotes")
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

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
            self.assertIn("live packet analysis", summaries[0]["long_resume"].lower())

    def test_stage_summarize_discards_incomplete_third_fragment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nDetailed markdown content for summarization.")

            short_calls = {"count": 0}

            def _short_execute(task: str) -> str:
                short_calls["count"] += 1
                if short_calls["count"] == 1:
                    return (
                        "A new TypeScript release improves inference for template-heavy codebases. "
                        "It matters because teams can remove custom workarounds and simplify internal tooling. "
                        "And this trailing fragment"
                    )
                return (
                    "A new TypeScript release improves inference for template-heavy codebases and targets teams that rely on complex template-driven abstractions in production applications. "
                    "It matters because engineers can remove brittle custom workarounds, simplify internal tooling, and trust the compiler in more advanced generic scenarios."
                )

            short_agent = SimpleNamespace(execute_task=_short_execute)
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "The new TypeScript release improves inference for template-heavy codebases. "
                    "The article explains how the update removes custom workarounds and simplifies internal tooling."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

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
                "A new TypeScript release improves inference for template-heavy codebases and targets teams that rely on complex template-driven abstractions in production applications. "
                "It matters because engineers can remove brittle custom workarounds, simplify internal tooling, and trust the compiler in more advanced generic scenarios."
            )
            self.assertEqual(short_calls["count"], 2)

    def test_stage_summarize_uses_title_for_second_sentence_when_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write("# Article\n\nDetailed markdown content for summarization.")

            short_agent = SimpleNamespace(
                execute_task=lambda task: "Improved database replication landed for edge clusters"
            )
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "Improved database replication landed for edge clusters. "
                    "The article describes the update that the database vendor shipped for edge deployments."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

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

    def test_stage_summarize_retries_short_summary_to_match_uniform_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\n"
                    "A database vendor introduced a new storage engine for analytical workloads.\n\n"
                    "The article explains performance goals, compatibility choices, and operational impact."
                )

            short_calls = {"count": 0}

            def _short_execute(task: str) -> str:
                short_calls["count"] += 1
                if short_calls["count"] == 1:
                    return "New storage engine launched. It improves analytics."
                return (
                    "A database vendor introduced a new storage engine aimed at analytical workloads and positioned it as a practical upgrade for existing deployments. "
                    "It matters because teams can improve query performance without abandoning familiar compatibility expectations or operational workflows."
                )

            short_agent = SimpleNamespace(execute_task=_short_execute)
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "A database vendor introduced a new storage engine for analytical workloads, framing it as an incremental but meaningful change for teams with growing query volumes. "
                    "The article explains how the release fits into the vendor's broader platform direction and where it is expected to deliver value first.\n\n"
                    "It covers performance goals, compatibility decisions, and the implementation tradeoffs behind the design, with particular attention to how the engine behaves under real operational constraints. "
                    "The technical discussion stays focused on the mechanics that practitioners would need to understand before deployment.\n\n"
                    "The article also points to migration and rollout considerations, noting that organizations still need to validate workload fit, operational assumptions, and upgrade sequencing before broad adoption. "
                    "That final framing keeps the piece grounded in execution details rather than treating the release as a purely promotional announcement."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

            summaries = pipeline.stage_summarize([{
                "title": "Database vendor ships analytics storage engine",
                "url": "https://example.com/storage-engine",
                "snippet": "A database vendor introduced a new storage engine for analytical workloads.",
                "local_path": article_path,
            }])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(short_calls["count"], 2)
            word_count = len(summaries[0]["summary"].split())
            self.assertGreaterEqual(word_count, newsletter_pipeline.SUMMARY_MIN_WORDS)
            self.assertLessEqual(word_count, newsletter_pipeline.SUMMARY_MAX_WORDS)

    def test_stage_summarize_regenerates_cached_summary_outside_uniform_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\n"
                    "A new orchestration platform release changes autoscaling and cluster recovery behavior.\n\n"
                    "The article explains the operational reasoning behind the release."
                )

            short_calls = {"count": 0}

            def _short_execute(task: str) -> str:
                short_calls["count"] += 1
                return (
                    "A new orchestration platform release changes autoscaling and cluster recovery behavior for distributed environments with stricter operational targets. "
                    "It matters because infrastructure teams can apply the update to improve resilience while keeping rollout planning tied to concrete production constraints."
                )

            short_agent = SimpleNamespace(execute_task=_short_execute)
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "A new orchestration platform release changes autoscaling and cluster recovery behavior, positioning the update as a practical operational improvement for distributed systems teams. "
                    "The article frames the release around day-to-day production concerns rather than around abstract feature marketing.\n\n"
                    "It explains how the new behavior affects scaling decisions, recovery workflows, and service continuity under failure conditions, with attention to implementation details that operators would need to evaluate. "
                    "The technical sections stay focused on mechanisms rather than on general positioning language.\n\n"
                    "The article closes with rollout considerations, including validation work, policy review, and production sequencing, so teams can judge where the update fits within existing infrastructure plans. "
                    "That practical framing helps translate the release into concrete operational decisions."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

            summaries = pipeline.stage_summarize([{
                "title": "Orchestration platform release",
                "url": "https://example.com/orchestration",
                "snippet": "A new orchestration platform release changes autoscaling and cluster recovery behavior.",
                "local_path": article_path,
                "summary": "Platform update landed. It matters.",
            }])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(short_calls["count"], 1)
            self.assertNotEqual(summaries[0]["summary"], "Platform update landed. It matters.")
            word_count = len(summaries[0]["summary"].split())
            self.assertGreaterEqual(word_count, newsletter_pipeline.SUMMARY_MIN_WORDS)
            self.assertLessEqual(word_count, newsletter_pipeline.SUMMARY_MAX_WORDS)

    def test_stage_summarize_retries_instead_of_trimming_extra_sentence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\n"
                    "A platform team released a new deployment controller for multi-region services.\n\n"
                    "The article covers rollout automation, safety controls, and operational impact."
                )

            short_calls = {"count": 0}

            def _short_execute(task: str) -> str:
                short_calls["count"] += 1
                if short_calls["count"] == 1:
                    return (
                        "A platform team released a new deployment controller for multi-region services with tighter automation around progressive delivery. "
                        "It matters because operators can reduce manual coordination during rollouts and recoveries across regions. "
                        "Teams can also centralize policy enforcement."
                    )
                return (
                    "A platform team released a new deployment controller for multi-region services with tighter automation around progressive delivery and recovery workflows. "
                    "It matters because operators can reduce manual rollout coordination while keeping safety controls and regional failover procedures aligned with production needs."
                )

            short_agent = SimpleNamespace(execute_task=_short_execute)
            long_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "A platform team released a new deployment controller for multi-region services, framing it as a practical update for organizations managing more complex production rollouts. "
                    "The article explains where the release fits into existing operational practices and why teams would adopt it.\n\n"
                    "It describes automation around rollout sequencing, regional coordination, and safety controls, with specific attention to how operators can manage progressive delivery and failure recovery. "
                    "The technical discussion stays focused on practical deployment mechanics.\n\n"
                    "The article also covers operational implications, including policy consistency, rollout guardrails, and the need to validate production procedures before adopting the controller broadly. "
                    "That final framing keeps the piece grounded in day-to-day operating concerns."
                )
            )
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

            summaries = pipeline.stage_summarize([{
                "title": "Deployment controller release",
                "url": "https://example.com/deployment-controller",
                "snippet": "A platform team released a new deployment controller for multi-region services.",
                "local_path": article_path,
            }])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(short_calls["count"], 2)
            self.assertEqual(
                summaries[0]["summary"],
                "A platform team released a new deployment controller for multi-region services with tighter automation around progressive delivery and recovery workflows. "
                "It matters because operators can reduce manual rollout coordination while keeping safety controls and regional failover procedures aligned with production needs."
            )
            self.assertNotIn("Teams can also centralize policy enforcement.", summaries[0]["summary"])

    def test_stage_summarize_retries_long_resume_to_match_uniform_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            article_path = os.path.join(tmpdir, "article.md")
            with open(article_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "# Article\n\n"
                    "A cloud vendor released a new orchestration update for edge deployments.\n\n"
                    "The update changes scheduling behavior, workload placement, and resilience controls.\n\n"
                    "The article also discusses rollout considerations, operational tradeoffs, and migration planning."
                )

            short_agent = SimpleNamespace(
                execute_task=lambda task: (
                    "A cloud vendor released a new orchestration update for edge deployments. "
                    "It changes scheduling behavior and resilience controls for operators."
                )
            )

            long_calls = {"count": 0}

            def _long_execute(task: str) -> str:
                long_calls["count"] += 1
                if long_calls["count"] == 1:
                    return (
                        "A cloud vendor released a new orchestration update for edge deployments.\n\n"
                        "It changes scheduling behavior and workload placement."
                    )
                return (
                    "A cloud vendor released a new orchestration update for edge deployments, focusing on how distributed workloads are scheduled and managed at the edge. "
                    "The article frames the release as a practical platform update rather than a research announcement, and it explains where the vendor expects teams to benefit first.\n\n"
                    "It details changes to scheduling behavior, workload placement logic, and resilience controls, with emphasis on how operators can tune placement decisions and maintain service continuity during failures. "
                    "The technical discussion stays close to deployment mechanics and the operational reasoning behind the new defaults.\n\n"
                    "The article also covers rollout considerations, tradeoffs, and migration planning, noting that teams still need to validate policies and capacity assumptions before broad adoption. "
                    "That closing guidance keeps the update grounded in day-to-day operating concerns instead of treating it as a purely feature-driven release."
                )

            long_agent = SimpleNamespace(execute_task=_long_execute)
            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            pipeline.manager = SimpleNamespace(
                get_agent=lambda name: short_agent if name == "summarizer_agent" else long_agent
            )

            summaries = pipeline.stage_summarize([{
                "title": "Edge orchestration update",
                "url": "https://example.com/edge-orchestration",
                "snippet": "A cloud vendor released a new orchestration update for edge deployments.",
                "local_path": article_path,
            }])

            self.assertEqual(len(summaries), 1)
            self.assertEqual(long_calls["count"], 2)
            paragraphs = [p for p in summaries[0]["long_resume"].split("\n\n") if p.strip()]
            self.assertEqual(len(paragraphs), 3)
            word_count = len(summaries[0]["long_resume"].split())
            self.assertGreaterEqual(word_count, 120)
            self.assertLessEqual(word_count, 180)

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
                "summary": "A valid short summary that already exists for this article and stays comfortably within the expected range. It explains why the update matters to engineering teams evaluating the change.",
                "long_resume": "",
                "quality_score": 7,
            }
            refilled_article = {**article, "long_resume": "Paragraph one.\n\nParagraph two.\n\nParagraph three."}
            calls = {"count": 0}

            def _stage_summarize(items):
                calls["count"] += 1
                return [refilled_article]

            pipeline.stage_summarize = _stage_summarize
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
                "summary": "A valid short summary that already exists for this article and stays comfortably within the expected range. It explains why the update matters to engineering teams evaluating the change.",
                "long_resume": "",
            }
            with open(summaries_path, "w", encoding="utf-8") as fh:
                json.dump([article], fh)

            pipeline = newsletter_pipeline.NewsletterPipeline.__new__(
                newsletter_pipeline.NewsletterPipeline
            )
            calls = {"count": 0}

            def _stage_summarize(items):
                calls["count"] += 1
                return [{**items[0], "long_resume": "Paragraph one.\n\nParagraph two.\n\nParagraph three."}]

            pipeline.stage_summarize = _stage_summarize
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
                    "summary": "Sentence one. Sentence two.",
                    "technologies": [],
                }])

            with open(output_path, encoding="utf-8") as fh:
                markdown = fh.read()

            self.assertIn("**URL:** https://altervista.example/newsletter#example-article", markdown)
            self.assertNotIn("**URL:** https://example.com/source-article", markdown)

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
