"""
Tests for benchmark_data.py — stats generation and data preparation utilities.
"""

import json
import logging
from pathlib import Path

import pytest

from benchmark_data import load_benchmark_data, make_stats

logger = logging.getLogger(__name__)


class TestMakeStats:
    def test_correct_counts(self, tmp_benchmark):
        logger.info("Testing make_stats on benchmark directory")
        stats = make_stats(tmp_benchmark)

        assert stats["total"] == 3
        assert stats["per_batch"]["batch-1"] == 2
        assert stats["per_batch"]["batch-2"] == 1
        logger.info("Stats: %s", stats)

    def test_missing_images_dir(self, tmp_path):
        """When no images/ directory exists, returns zeros."""
        logger.info("Testing make_stats with missing images dir")
        stats = make_stats(tmp_path)
        assert stats == {"per_batch": {}, "total": 0}

    def test_empty_images_dir(self, tmp_path):
        """An empty images/ directory → zero total."""
        (tmp_path / "images").mkdir()
        stats = make_stats(tmp_path)
        assert stats["total"] == 0
        assert stats["per_batch"] == {}


class TestLoadBenchmarkData:
    def test_loads_csv(self, tmp_benchmark):
        logger.info("Testing load_benchmark_data")
        df = load_benchmark_data(tmp_benchmark / "benchmark.csv")
        assert len(df) == 3
        assert "image_name" in df.columns
        assert "transcript" in df.columns
        logger.info("Loaded %d rows from benchmark CSV", len(df))
