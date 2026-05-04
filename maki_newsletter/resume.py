"""
Maki Newsletter — resume from stage 3.

Usage (from the repo root):
    python -m maki_newsletter.resume [--trends]

Options:
    --trends    Re-fetch trend signals before ranking.

The command loads the incremental manifest written by stage_download during a
previous full pipeline run and executes stages 3–6 (read → rank → summarize →
evaluate).  It always targets the current ISO week, so downloaded articles must
already be present in the articles directory for that week.
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("maki_newsletter")

from maki_newsletter.pipeline import NewsletterPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the newsletter pipeline from stage 3 "
            "(read → rank → summarize → evaluate) using the incremental manifest."
        )
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Re-fetch trend signals before ranking (adds network calls).",
    )
    args = parser.parse_args()

    logger.info("Maki Newsletter — resuming from stage 3 …")
    logger.info("Working directory: %s", os.getcwd())

    try:
        pipeline = NewsletterPipeline()
        eval_path = pipeline.run_from_stage3(refetch_trends=args.trends)
        print(
            f"\nEvaluation file written:\n  {eval_path}\n\n"
            "Review the file, then generate the final newsletter:\n"
            "  python -m maki_newsletter.generate\n"
        )
    except RuntimeError as exc:
        logger.error("Resume failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
