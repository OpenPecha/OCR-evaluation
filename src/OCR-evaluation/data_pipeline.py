"""
Data pipeline helpers for the OCR evaluation project.

- Reading benchmark image paths
- Writing per-model inference results to CSV
"""

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_image_directory(benchmark_path: Path) -> Path:
    """Return the images/ sub-directory of the benchmark folder."""
    return benchmark_path / "images"


def get_all_image_paths(
    benchmark_path: Path,
    batch_id: str | None = None,
) -> List[Tuple[str, str, Path]]:
    """
    Collect benchmark images and return a flat list of
    ``(image_name, batch_id, image_path)`` tuples, sorted by batch then name.

    Parameters
    ----------
    benchmark_path : Path
        Root of the benchmark data directory.
    batch_id : str or None
        If given, only images from this batch are returned.
        If ``None`` (default), images from every batch are collected.
    """
    images_dir = get_image_directory(benchmark_path)
    logger.debug("Scanning benchmark images in %s", images_dir)

    if not images_dir.exists():
        logger.warning("Images directory does not exist: %s", images_dir)
        return []

    results: List[Tuple[str, str, Path]] = []
    for batch_dir in sorted(images_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        bid = batch_dir.name
        if batch_id is not None and bid != batch_id:
            continue
        batch_count = 0
        for img_file in sorted(batch_dir.iterdir()):
            if img_file.is_file():
                results.append((img_file.name, bid, img_file))
                batch_count += 1
        logger.debug("  Batch %-12s  %d images", bid, batch_count)

    logger.info("Collected %d images from %d batches", len(results),
                len({r[1] for r in results}))
    return results


def write_model_output(
    model_name: str,
    results: List[Tuple[str, str, str]],
    output_dir: Path,
) -> Path:
    """
    Write OCR inference results to ``output_dir/{model_name}.csv``.

    Uses :mod:`csv.writer` so that double-quote characters inside
    inference text are properly escaped (``""``), avoiding parse
    errors when the CSV is read back with :func:`pandas.read_csv`.

    Parameters
    ----------
    model_name : str
        Name of the OCR model (used as the CSV filename stem).
    results : list of (image_name, batch_id, inference_text)
        One entry per image.
    output_dir : Path
        Directory in which to create the CSV.

    Returns
    -------
    Path
        The path of the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_name}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "batch_id", "inference"])
        for image_name, batch_id, inference_text in results:
            writer.writerow([image_name, batch_id, inference_text])

    logger.info("Wrote %d inference rows → %s", len(results), csv_path)
    return csv_path


def reformat_model_csvs(output_dir: Path) -> None:
    """
    Re-read every model CSV in *output_dir* and re-write it using
    :mod:`csv.writer` for consistent quoting (matching the benchmark CSV).

    This avoids having to re-run inference just to fix CSV formatting.
    """
    for csv_path in sorted(output_dir.glob("*.csv")):
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(df.columns))
            for _, row in df.iterrows():
                inference = row.get("inference", "")
                if pd.isna(inference):
                    inference = ""
                writer.writerow([
                    row["image_name"],
                    row["batch_id"],
                    inference,
                ])
        logger.info("Reformatted %s (%d rows)", csv_path, len(df))
    logger.info("All model CSVs in %s reformatted.", output_dir)


if __name__ == "__main__":
    """Reformat existing model CSVs with proper csv.writer quoting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )
    from eval_config import OUTPUT_DIR
    reformat_model_csvs(OUTPUT_DIR)
