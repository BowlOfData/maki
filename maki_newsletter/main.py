"""
Maki Newsletter — entry point.

Usage (from the repo root):
    python -m maki_newsletter.main

Or directly:
    python maki_newsletter/main.py

The script changes the working directory to the maki_newsletter/ folder so that
all relative paths (output/, output/articles/) resolve correctly regardless of
where the script is invoked from.
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Set CWD to the maki_newsletter/ directory so that all relative paths in
# plugins (FileWriter, WebToMd) resolve correctly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

# Ensure the repo root is on sys.path so that `import maki` works when the
# script is run directly (python maki_newsletter/main.py).
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Logging — human-readable output to stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)

logger = logging.getLogger("maki_newsletter")

# ---------------------------------------------------------------------------
# Import pipeline (after path setup)
# ---------------------------------------------------------------------------
from maki_newsletter.pipeline import NewsletterPipeline  # noqa: E402


def main() -> None:
    logger.info("Maki Newsletter generator starting …")
    logger.info("Working directory: %s", os.getcwd())

    try:
        pipeline = NewsletterPipeline()
        eval_path = pipeline.run()
        print(
            f"\nEvaluation file written:\n  {eval_path}\n\n"
            "Review the file, then generate the final newsletter:\n"
            "  python -m maki_newsletter.generate\n"
        )
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
