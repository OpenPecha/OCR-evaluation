"""
Benchmark data preparation utilities.

Copies images and transcripts into the standard benchmark layout and
generates ``benchmark.csv`` and ``stats.json``.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_benchmark_data(data_path: Path) -> pd.DataFrame:
    """Load the benchmark data from the given path."""
    logger.info("Loading benchmark data from %s", data_path)
    return pd.read_csv(data_path)


def copy_images_to_destination(finalised_data: pd.DataFrame, source_path: Path):
    """Copy images and transcripts into the benchmark directory structure."""
    copied = 0
    for _index, row in finalised_data.iterrows():
        image_name = row['task_name']
        batch_id = row['batch_id']
        transcript = row['task_transcript']
        if batch_id == "b1":
            batch_id = "batch-1"
        if batch_id == "batch-05":
            batch_id = "batch-5"
        source_image_path = source_path / batch_id / "images" / image_name
        destination_image_path = Path(f"./data/benchmark/images/{batch_id}/{image_name}")
        destination_transcript_path = Path(f"./data/benchmark/transcripts/{batch_id}/{Path(image_name).stem}.txt")
        destination_image_path.parent.mkdir(parents=True, exist_ok=True)
        destination_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_image_path, destination_image_path)
        with open(destination_transcript_path, "w") as f:
            f.write(transcript)
        copied += 1

    logger.info("Copied %d images and transcripts from %s", copied, source_path)


def make_benchmark_csv(benchmark_path: Path):
    """Generate ``benchmark.csv`` from images and transcripts on disk.

    Uses :mod:`csv.writer` so that double-quote characters inside
    transcript text are properly escaped (``""``), avoiding parse
    errors when the CSV is read back with :func:`pandas.read_csv`.
    """
    import csv

    benchmark_csv_path = benchmark_path / "benchmark.csv"
    count = 0
    with open(benchmark_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "batch_id", "transcript"])
        for batch_id in sorted(os.listdir(benchmark_path / "images")):
            batch_dir = benchmark_path / "images" / batch_id
            if not batch_dir.is_dir():
                continue
            for image in sorted(os.listdir(batch_dir)):
                image_name = image.split("/")[-1]
                transcript_path = (
                    benchmark_path / "transcripts" / batch_id
                    / f"{Path(image_name).stem}.txt"
                )
                transcript = transcript_path.read_text(encoding="utf-8")
                writer.writerow([image_name, batch_id, transcript])
                count += 1

    logger.info("Wrote benchmark CSV with %d entries → %s", count, benchmark_csv_path)


def make_stats(benchmark_path: Path) -> dict:
    """
    Get total images per batch and total images overall.
    Returns a dict with "per_batch" (batch_id -> count) and "total" (int).
    """
    images_dir = benchmark_path / "images"
    if not images_dir.exists():
        logger.warning("Images directory does not exist: %s", images_dir)
        return {"per_batch": {}, "total": 0}
    per_batch = {}
    for batch_id in sorted(os.listdir(images_dir)):
        batch_dir = images_dir / batch_id
        if batch_dir.is_dir():
            count = sum(1 for _ in batch_dir.iterdir() if _.is_file())
            per_batch[batch_id] = count
            logger.debug("  Batch %-12s  %d images", batch_id, count)

    total = sum(per_batch.values())
    logger.info("Stats: %d batches, %d total images", len(per_batch), total)
    return {"per_batch": per_batch, "total": total}


def main():
    """Main function to run the script."""
    logger.info("Generating benchmark stats …")
    stats = make_stats(Path("./data/benchmark/"))
    with open("./data/benchmark/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats written to data/benchmark/stats.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )
    main()