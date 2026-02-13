"""
General-purpose file I/O utilities.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def read_csv_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    logger.debug("Reading CSV: %s", file_path)
    return pd.read_csv(file_path)


def write_csv_file(file_path: Path, data: pd.DataFrame) -> None:
    """Write a DataFrame to a CSV file."""
    data.to_csv(file_path, index=False)
    logger.debug("Wrote CSV: %s (%d rows)", file_path, len(data))


def read_json_file(file_path: Path) -> dict:
    """Read a JSON file and return a dict."""
    logger.debug("Reading JSON: %s", file_path)
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json_file(file_path: Path, data: dict) -> None:
    """Write a dict to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.debug("Wrote JSON: %s", file_path)


def read_image_file(file_path: Path) -> Image.Image:
    """Read an image file and return a PIL Image."""
    logger.debug("Reading image: %s", file_path)
    return Image.open(file_path)
