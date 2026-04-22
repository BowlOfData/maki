"""
Maki Newsletter — final newsletter generator.

Run this after reviewing and editing the summaries JSON produced by the
main pipeline:

    python -m maki_newsletter.generate

All articles present in the most recent summaries_<week>_<year>.json are
included in the newsletter — remove any entries from that file before running
this script if you want to exclude them.
"""

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

logger = logging.getLogger("maki_newsletter.generate")

from maki_newsletter.pipeline import NewsletterPipeline  # noqa: E402


def main() -> None:
    logger.info("Newsletter generator starting …")

    try:
        pipeline = NewsletterPipeline()
        output_path = pipeline.run_generate()
        print(f"\nNewsletter generated successfully:\n  {output_path}\n")
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
