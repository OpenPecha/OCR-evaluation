"""
Character Error Rate (CER) evaluation for OCR models.

Compares OCR inference output against ground-truth transcripts and computes
CER at per-image, per-batch, and overall levels.

CER = edit_distance(inference, reference) / len(reference)

Usage
-----
    python CER_evaluation.py                 # evaluate all models
    python CER_evaluation.py --model Modern  # evaluate a single model
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from eval_config import BENCHMARK_CSV, OUTPUT_DIR, PROJECT_ROOT
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Output paths ─────────────────────────────────────────────────────────────
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"


# ── Levenshtein edit distance ────────────────────────────────────────────────
def _edit_distance(s1: str, s2: str) -> int:
    """
    Compute the character-level Levenshtein edit distance between *s1* and *s2*
    using the standard dynamic-programming algorithm (O(n·m) time and O(min(n,m))
    space via a single-row optimisation).
    """
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    # s1 is now the longer (or equal) string
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr
    return prev[-1]


def compute_cer(inference: str, reference: str) -> float:
    """
    Compute Character Error Rate.

    Returns 0.0 when both strings are empty.
    Returns 1.0 when the reference is empty but inference is not (by convention).
    """
    if len(reference) == 0:
        return 0.0 if len(inference) == 0 else 1.0
    return _edit_distance(inference, reference) / len(reference)


# ── Normalisation ────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """
    Minimal text normalisation applied to *both* reference and inference before
    CER computation so that trivial whitespace differences don't dominate the
    error rate.

    * Strip leading / trailing whitespace.
    * Collapse runs of whitespace to a single space.
    """
    return " ".join(text.split())


# ── Loading helpers ──────────────────────────────────────────────────────────
def load_benchmark(benchmark_csv: Path) -> Dict[Tuple[str, str], str]:
    """
    Load the ground-truth benchmark CSV and return a dict mapping
    ``(image_name, batch_id)`` → ``transcript``.
    """
    df = pd.read_csv(benchmark_csv, dtype=str)
    # Strip column names and values of stray whitespace
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if hasattr(df[col], "str"):
            df[col] = df[col].str.strip()
    lookup: Dict[Tuple[str, str], str] = {}
    for _, row in df.iterrows():
        key = (row["image_name"], row["batch_id"])
        lookup[key] = row.get("transcript", "")
    return lookup


def load_model_csv(model_csv: Path) -> List[Tuple[str, str, str]]:
    """
    Load a model inference CSV and return a list of
    ``(image_name, batch_id, inference_text)`` tuples.
    """
    df = pd.read_csv(model_csv, dtype=str)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if hasattr(df[col], "str"):
            df[col] = df[col].str.strip()
    rows = []
    for _, row in df.iterrows():
        rows.append((
            row["image_name"],
            row["batch_id"],
            row.get("inference", ""),
        ))
    return rows


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    model_csv: Path,
    benchmark: Dict[Tuple[str, str], str],
) -> Dict:
    """
    Evaluate a single model against the benchmark.

    Returns a dict with:
    - ``model``: model name
    - ``overall_cer``: macro-average CER over all images
    - ``per_batch``: ``{batch_id: {"cer": float, "count": int}}``
    - ``per_image``: list of ``{"image_name", "batch_id", "cer"}``
    """
    inferences = load_model_csv(model_csv)

    per_image: List[Dict] = []
    raw_cers: List[float] = []               # keep unrounded for aggregation
    batch_accum: Dict[str, List[float]] = {}

    for image_name, batch_id, inference_text in inferences:
        reference = benchmark.get((image_name, batch_id), "")
        cer = compute_cer(
            _normalise(inference_text if pd.notna(inference_text) else ""),
            _normalise(reference if pd.notna(reference) else ""),
        )
        per_image.append({
            "image_name": image_name,
            "batch_id": batch_id,
            "cer": round(cer, 6),
        })
        raw_cers.append(cer)
        batch_accum.setdefault(batch_id, []).append(cer)

    # Per-batch aggregates
    per_batch = {}
    for bid in sorted(batch_accum):
        vals = batch_accum[bid]
        per_batch[bid] = {
            "cer": round(sum(vals) / len(vals), 6),
            "count": len(vals),
        }

    # Overall (computed from raw unrounded values for precision)
    overall_cer = sum(raw_cers) / len(raw_cers) if raw_cers else 0.0

    return {
        "model": model_name,
        "overall_cer": round(overall_cer, 6),
        "per_batch": per_batch,
        "per_image": per_image,
    }


def save_evaluation(result: Dict, output_dir: Path) -> Tuple[Path, Path]:
    """
    Persist evaluation results.

    Writes:
    - ``{output_dir}/{model}_cer.json``  – full results (overall + per-batch + per-image)
    - ``{output_dir}/{model}_cer.csv``   – per-image CER for easy inspection

    Returns the paths of both files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = result["model"]

    json_path = output_dir / f"{model}_cer.json"
    csv_path = output_dir / f"{model}_cer.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(result["per_image"])
    df.to_csv(csv_path, index=False)

    return json_path, csv_path


def save_summary(all_results: List[Dict], output_dir: Path) -> Path:
    """
    Write a summary CSV comparing overall and per-batch CER across all models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"

    rows = []
    for result in all_results:
        row = {"model": result["model"], "overall_cer": result["overall_cer"]}
        for bid, stats in result["per_batch"].items():
            row[f"cer_{bid}"] = stats["cer"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns: model, overall, then batch cols sorted
    batch_cols = sorted([c for c in df.columns if c.startswith("cer_")])
    df = df[["model", "overall_cer"] + batch_cols]
    df.to_csv(summary_path, index=False)

    return summary_path


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR model(s) by computing Character Error Rate."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate a single model (by name). If omitted, all models in "
             "data/models/ are evaluated.",
    )
    args = parser.parse_args()

    # Load ground-truth benchmark
    logger.info("Loading benchmark from %s", BENCHMARK_CSV)
    benchmark = load_benchmark(BENCHMARK_CSV)
    logger.info("Benchmark contains %d entries", len(benchmark))

    # Discover model CSVs
    if args.model:
        csv_path = OUTPUT_DIR / f"{args.model}.csv"
        if not csv_path.exists():
            logger.error("Model CSV not found: %s", csv_path)
            return
        model_csvs = [(args.model, csv_path)]
    else:
        model_csvs = [
            (p.stem, p)
            for p in sorted(OUTPUT_DIR.glob("*.csv"))
            if p.is_file()
        ]

    if not model_csvs:
        logger.error("No model CSVs found in %s", OUTPUT_DIR)
        return

    # Evaluate each model
    all_results: List[Dict] = []
    for model_name, csv_path in model_csvs:
        logger.info("Evaluating model: %s", model_name)
        result = evaluate_model(model_name, csv_path, benchmark)
        json_path, cer_csv = save_evaluation(result, EVALUATION_DIR)
        all_results.append(result)
        logger.info(
            "  Overall CER: %.4f  →  %s", result["overall_cer"], json_path
        )
        for bid, stats in sorted(result["per_batch"].items()):
            logger.info(
                "    %s: CER=%.4f  (%d images)", bid, stats["cer"], stats["count"]
            )

    # Cross-model summary
    if len(all_results) > 1:
        summary_path = save_summary(all_results, EVALUATION_DIR)
        logger.info("Summary written to %s", summary_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
