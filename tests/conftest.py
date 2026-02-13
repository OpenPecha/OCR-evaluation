"""
Shared pytest fixtures for the OCR evaluation test suite.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

# ── Make the source packages importable ──────────────────────────────────────
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "OCR-evaluation"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DATA_PREP_DIR = Path(__file__).resolve().parents[1] / "src" / "data-preparations"
if str(_DATA_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP_DIR))


# ── Configure logging for tests ─────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _setup_test_logging(caplog):
    """Capture all log output during tests so it shows in the terminal."""
    with caplog.at_level(logging.DEBUG):
        yield


# ── Temporary benchmark directory ────────────────────────────────────────────
@pytest.fixture
def tmp_benchmark(tmp_path):
    """
    Create a minimal benchmark directory structure:

        tmp_path/
            images/
                batch-1/
                    img_001.txt   (dummy file standing in for an image)
                    img_002.txt
                batch-2/
                    img_003.txt
            transcripts/
                batch-1/
                    img_001.txt
                    img_002.txt
                batch-2/
                    img_003.txt
            benchmark.csv
    """
    images_dir = tmp_path / "images"
    transcripts_dir = tmp_path / "transcripts"

    # batch-1: 2 images
    (images_dir / "batch-1").mkdir(parents=True)
    (images_dir / "batch-1" / "img_001.png").write_text("fake-image-data-1")
    (images_dir / "batch-1" / "img_002.png").write_text("fake-image-data-2")

    (transcripts_dir / "batch-1").mkdir(parents=True)
    (transcripts_dir / "batch-1" / "img_001.txt").write_text("བཀྲ་ཤིས་བདེ་ལེགས")
    (transcripts_dir / "batch-1" / "img_002.txt").write_text("ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ")

    # batch-2: 1 image
    (images_dir / "batch-2").mkdir(parents=True)
    (images_dir / "batch-2" / "img_003.png").write_text("fake-image-data-3")

    (transcripts_dir / "batch-2").mkdir(parents=True)
    (transcripts_dir / "batch-2" / "img_003.txt").write_text("སྤྱན་རས་གཟིགས")

    # benchmark.csv
    df = pd.DataFrame([
        {"image_name": "img_001.png", "batch_id": "batch-1", "transcript": "བཀྲ་ཤིས་བདེ་ལེགས"},
        {"image_name": "img_002.png", "batch_id": "batch-1", "transcript": "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ"},
        {"image_name": "img_003.png", "batch_id": "batch-2", "transcript": "སྤྱན་རས་གཟིགས"},
    ])
    df.to_csv(tmp_path / "benchmark.csv", index=False)

    return tmp_path


# ── Temporary output directory ───────────────────────────────────────────────
@pytest.fixture
def tmp_output(tmp_path):
    """Return a clean temporary directory for output files."""
    out = tmp_path / "output"
    out.mkdir()
    return out
