#!/usr/bin/env python3
"""
Test script to verify logging functionality in Maki (placeholder).

The original test relied on the Maki class which has been removed.
Logging configuration is covered by configure_logging() in maki/logging_config.py.
"""

import logging
import unittest

from maki.logging_config import configure_logging


class TestLoggingConfig(unittest.TestCase):

    def test_configure_logging_does_not_raise(self):
        """configure_logging() must be callable without error."""
        try:
            configure_logging()
        except Exception as exc:
            self.fail(f"configure_logging() raised an unexpected exception: {exc}")

    def test_configure_logging_sets_up_logger(self):
        """After configure_logging(), the 'maki' logger should exist."""
        configure_logging()
        logger = logging.getLogger("maki")
        self.assertIsNotNone(logger)


if __name__ == "__main__":
    unittest.main()
