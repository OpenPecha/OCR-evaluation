"""
Tests for utils.py — CSV, JSON, and image file I/O helpers.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from utils import read_csv_file, read_json_file, write_csv_file, write_json_file

logger = logging.getLogger(__name__)


class TestCSVIO:
    def test_roundtrip(self, tmp_path):
        logger.info("Testing CSV read/write round-trip")
        csv_path = tmp_path / "test.csv"
        original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_csv_file(csv_path, original)
        loaded = read_csv_file(csv_path)

        pd.testing.assert_frame_equal(original, loaded)
        logger.info("CSV round-trip verified: %s", csv_path)

    def test_write_creates_file(self, tmp_path):
        csv_path = tmp_path / "new.csv"
        df = pd.DataFrame({"col": [1]})
        write_csv_file(csv_path, df)
        assert csv_path.exists()

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_csv_file(tmp_path / "nonexistent.csv")


class TestJSONIO:
    def test_roundtrip(self, tmp_path):
        logger.info("Testing JSON read/write round-trip")
        json_path = tmp_path / "test.json"
        original = {"key": "value", "nested": {"a": 1}, "list": [1, 2, 3]}

        write_json_file(json_path, original)
        loaded = read_json_file(json_path)

        assert loaded == original
        logger.info("JSON round-trip verified: %s", json_path)

    def test_unicode_preservation(self, tmp_path):
        """Tibetan characters survive a JSON round-trip."""
        logger.info("Testing JSON Unicode (Tibetan) preservation")
        json_path = tmp_path / "tibetan.json"
        data = {"text": "བཀྲ་ཤིས་བདེ་ལེགས"}

        write_json_file(json_path, data)
        loaded = read_json_file(json_path)

        assert loaded["text"] == "བཀྲ་ཤིས་བདེ་ལེགས"
        logger.info("Tibetan text preserved through JSON round-trip")

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_json_file(tmp_path / "nonexistent.json")
