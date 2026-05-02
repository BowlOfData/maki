import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maki_newsletter import publish as newsletter_publish


class TestNewsletterPublish(unittest.TestCase):

    def test_trim_resume_to_word_limit_preserves_complete_sentences(self):
        text = (
            f"{'alpha ' * 260}."
            f" {'beta ' * 260}."
            " gamma tail."
        )

        trimmed = newsletter_publish._trim_resume_to_word_limit(text, max_words=500)

        self.assertLessEqual(len(trimmed.split()), 500)
        self.assertTrue(trimmed.endswith("."))
        self.assertNotIn("gamma tail", trimmed)

    def test_build_professional_resume_uses_agent_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "article.md"
            md_path.write_text(
                "# Edge database update\n\n"
                "The article explains a new replication release for edge clusters.\n\n"
                "It covers failover behavior, consistency tradeoffs, and deployment constraints.",
                encoding="utf-8",
            )

            captured = {}

            def _execute(task: str) -> str:
                captured["task"] = task
                return (
                    "The vendor released a new replication update for edge database clusters.\n\n"
                    "The article explains how the release changes failover behavior, consistency tradeoffs, "
                    "and rollout constraints for distributed deployments."
                )

            with patch.object(
                newsletter_publish,
                "_get_resume_agent",
                return_value=SimpleNamespace(execute_task=_execute),
            ):
                resume = newsletter_publish._build_professional_resume(md_path)

            self.assertIn("no more than 500 words", captured["task"])
            self.assertIn("2 to 4 short paragraphs", captured["task"])
            self.assertIn("Edge database update", captured["task"])
            self.assertIn("replication update", resume.lower())
            self.assertNotIn("bullet points", resume.lower())

    def test_build_professional_resume_falls_back_to_article_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "article.md"
            md_path.write_text(
                "# Kernel release\n\n"
                "Kernel maintainers published a release candidate with scheduler changes.\n\n"
                "The article describes performance benchmarks and compatibility caveats for operators.",
                encoding="utf-8",
            )

            with patch.object(
                newsletter_publish,
                "_get_resume_agent",
                return_value=SimpleNamespace(execute_task=lambda task: ""),
            ):
                resume = newsletter_publish._build_professional_resume(md_path)

            self.assertIn("scheduler changes", resume.lower())
            self.assertIn("performance benchmarks", resume.lower())


if __name__ == "__main__":
    unittest.main()
