"""
Tests for logging_config.py — verify the centralized logging setup.
"""

import logging
from pathlib import Path

import pytest

from logging_config import LOG_DIR, setup_logging


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        """setup_logging should create the log directory if it doesn't exist."""
        log_file = tmp_path / "logs" / "test.log"

        # Clear root handlers so setup_logging can add new ones
        root = logging.getLogger()
        root.handlers.clear()

        setup_logging(log_file=log_file)

        assert log_file.parent.exists()
        assert log_file.exists()

        # Clean up
        root.handlers.clear()

    def test_idempotent(self, tmp_path):
        """Calling setup_logging twice should not duplicate handlers."""
        log_file = tmp_path / "logs" / "test.log"

        root = logging.getLogger()
        root.handlers.clear()

        setup_logging(log_file=log_file)
        count_after_first = len(root.handlers)

        setup_logging(log_file=log_file)
        count_after_second = len(root.handlers)

        assert count_after_first == count_after_second

        # Clean up
        root.handlers.clear()

    def test_log_message_written_to_file(self, tmp_path):
        """Messages logged after setup should appear in the log file."""
        log_file = tmp_path / "logs" / "test.log"

        root = logging.getLogger()
        root.handlers.clear()

        setup_logging(log_file=log_file)

        test_logger = logging.getLogger("test.module")
        test_logger.info("Hello from test")

        # Flush handlers
        for handler in root.handlers:
            handler.flush()

        log_content = log_file.read_text()
        assert "Hello from test" in log_content
        assert "test.module" in log_content

        # Clean up
        root.handlers.clear()
