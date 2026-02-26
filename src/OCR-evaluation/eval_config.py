"""
Configuration for the OCR evaluation pipeline.

Defines paths to the tibetan-ocr-app, OCR models, line detection model,
benchmark data, output directory, and per-model OCR settings.

Every model entry in MODELS_CONFIG must include a ``"provider"`` key that
tells the pipeline which OCR backend to use:

    - ``"bdrc"``               → local BDRC OCRPipeline
    - ``"google_cloud_vision"`` → Google Cloud Vision API
    - ``"gemini"``             → Google Gemini API
    - ``"deepseek"``           → DeepSeek vision API

Only BDRC models require the line-detection model and local model weights.
API-based models need credentials via environment variables.
"""

import os
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

# ── Raw Google Vision API responses (gzipped JSON per image) ─────────────────
GOOGLE_VISION_RAW_DIR = PROJECT_ROOT / "data" / "google_vision_raw"

# ── Raw Gemini OCR text output (one .txt per image) ─────────────────────────
GEMINI_RAW_DIR = PROJECT_ROOT / "data" / "gemini_raw"

# ── Per-model OCR settings ───────────────────────────────────────────────────
# Each key becomes the output CSV filename: data/models/{key}.csv
#
# "provider" determines which OCR backend is used:
#   - "bdrc"                → local BDRC model (requires model_dir + line detection)
#   - "google_cloud_vision" → Google Cloud Vision TEXT_DETECTION API
#   - "gemini"              → Google Gemini vision model
#   - "deepseek"            → DeepSeek vision model (OpenAI-compatible API)
MODELS_CONFIG = {
    # ── BDRC local models ────────────────────────────────────────────────
    "Ume_Druma": {
        "provider": "bdrc",
        "model_dir": str(OCR_MODELS_DIR / "Ume_Druma"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
    "Ume_Petsuk": {
        "provider": "bdrc",
        "model_dir": str(OCR_MODELS_DIR / "Ume_Petsuk"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
    "Woodblock": {
        "provider": "bdrc",
        "model_dir": str(OCR_MODELS_DIR / "Woodblock"),
        "k_factor": 2.5,
        "bbox_tolerance": 4.0,
        "encoding": "unicode",
        "merge_lines": True,
        "dewarp": False,
    },
    # ── API-based models ────────────────────────────────────────────────
    "Google_Vision": {
        "provider": "google_cloud_vision",
        "max_workers": 4,
    },
    # ── Gemini models ────────────────────────────────────────────────────
    "Gemini_2_Flash": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-2.0-flash",
    },
    "Gemini_2.5_Flash": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-2.5-flash",
    },
    "Gemini_3_Flash": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-3-flash-preview",
    },
    "Gemini_3_Pro": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-3-pro-preview",
    },
    "Gemini_Prompt_3_Flash": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-3-flash-preview",
        "use_structured_prompt": True,
    },
    # "DeepSeek": {
    #     "provider": "deepseek",
    #     "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    #     "base_url": "https://api.deepseek.com",
    #     "model_name": "deepseek-chat",
    # },
}
