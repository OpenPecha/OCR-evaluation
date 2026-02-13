"""
Tests for data_pipeline.py — image path collection and model output writing.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline import get_all_image_paths, get_image_directory, write_model_output

logger = logging.getLogger(__name__)


class TestGetImageDirectory:
    def test_returns_images_subdir(self, tmp_path):
        expected = tmp_path / "images"
        assert get_image_directory(tmp_path) == expected


class TestGetAllImagePaths:
    def test_collects_images_from_benchmark(self, tmp_benchmark):
        logger.info("Testing image path collection from benchmark dir")
        images = get_all_image_paths(tmp_benchmark)

        assert len(images) == 3

        # Verify structure: (image_name, batch_id, image_path)
        names = [name for name, _batch, _path in images]
        batches = [batch for _name, batch, _path in images]

        assert "img_001.png" in names
        assert "img_002.png" in names
        assert "img_003.png" in names
        assert "batch-1" in batches
        assert "batch-2" in batches

        # All paths should exist
        for _name, _batch, path in images:
            assert path.exists()

        logger.info("Collected %d image paths across batches: %s",
                     len(images), sorted(set(batches)))

    def test_empty_benchmark(self, tmp_path):
        """When no images directory exists, return an empty list."""
        logger.info("Testing with empty benchmark (no images dir)")
        images = get_all_image_paths(tmp_path)
        assert images == []

    def test_ignores_files_in_images_root(self, tmp_path):
        """Files directly in images/ (not in a batch subdir) are ignored."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "stray_file.txt").write_text("should be ignored")

        images = get_all_image_paths(tmp_path)
        assert images == []

    def test_sorted_output(self, tmp_benchmark):
        """Results should be sorted by batch then image name."""
        images = get_all_image_paths(tmp_benchmark)
        batch_name_pairs = [(batch, name) for name, batch, _path in images]
        assert batch_name_pairs == sorted(batch_name_pairs)
        logger.info("Sort order verified: %s", batch_name_pairs)


class TestWriteModelOutput:
    def test_writes_csv_with_correct_content(self, tmp_output):
        logger.info("Testing write_model_output")
        results = [
            ("img_001.png", "batch-1", "recognized text 1"),
            ("img_002.png", "batch-1", "recognized text 2"),
            ("img_003.png", "batch-2", "recognized text 3"),
        ]
        csv_path = write_model_output("test_model", results, tmp_output)

        assert csv_path.exists()
        assert csv_path.name == "test_model.csv"

        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert list(df.columns) == ["image_name", "batch_id", "inference"]
        assert df.iloc[0]["inference"] == "recognized text 1"
        logger.info("Model output CSV verified: %s", csv_path)

    def test_creates_output_directory(self, tmp_path):
        """Output dir is created automatically if it doesn't exist."""
        out_dir = tmp_path / "new" / "nested" / "dir"
        results = [("a.png", "b1", "text")]

        csv_path = write_model_output("m", results, out_dir)
        assert csv_path.exists()
        logger.info("Auto-created output directory: %s", out_dir)

    def test_empty_results(self, tmp_output):
        """Writing an empty result list should still produce a valid CSV."""
        csv_path = write_model_output("empty", [], tmp_output)
        df = pd.read_csv(csv_path)
        assert len(df) == 0
        assert list(df.columns) == ["image_name", "batch_id", "inference"]
