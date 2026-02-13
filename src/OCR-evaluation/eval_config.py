"""
Configuration for the OCR evaluation pipeline.

Defines paths to the tibetan-ocr-app, OCR models, line detection model,
benchmark data, output directory, and per-model OCR settings.
"""

from pathlib import Path

# ── Project root (two levels up from this file) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── tibetan-ocr-app clone ────────────────────────────────────────────────────
TIBETAN_OCR_APP_DIR = PROJECT_ROOT / "vendor" / "tibetan-ocr-app"

# ── OCR model directory (downloaded from GitHub Releases) ────────────────────
OCR_MODELS_DIR = TIBETAN_OCR_APP_DIR / "OCRModels"

# ── Line detection model (shipped via Git LFS inside the cloned repo) ────────
LINE_DETECTION_MODEL = str(
    TIBETAN_OCR_APP_DIR / "Models" / "Lines" / "PhotiLines.onnx"
)
LINE_DETECTION_PATCH_SIZE = 512

# ── Concurrency ───────────────────────────────────────────────────────────────
# Number of background threads used for OCR inference (tune to your machine).
NUM_WORKERS = 4

# ── Benchmark data ───────────────────────────────────────────────────────────
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
BENCHMARK_CSV = BENCHMARK_DIR / "benchmark.csv"

# ── Output directory for per-model inference CSVs ────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "models"

# ── Per-model OCR settings ───────────────────────────────────────────────────
# Each key becomes the output CSV filename: data/models/{key}.csv
# "model_dir" is relative to OCR_MODELS_DIR.
MODELS_CONFIG = {
    "Ume_Druma": {
        "model_dir": str(OCR_MODELS_DIR / "Ume_Druma"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
    "Ume_Petsuk": {
        "model_dir": str(OCR_MODELS_DIR / "Ume_Petsuk"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
    "Woodblock": {
        "model_dir": str(OCR_MODELS_DIR / "Woodblock"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
}
