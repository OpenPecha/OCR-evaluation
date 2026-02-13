"""
Main OCR evaluation pipeline.

For every model listed in eval_config.MODELS_CONFIG (or a single model
selected via ``--model``):
  1. Detect the provider (bdrc, google_cloud_vision, gemini, deepseek).
  2. Initialise only the backend that the model requires.
  3. Run inference on every benchmark image (optionally filtered by ``--batch``).
  4. Write results to data/models/{model_name}.csv.

Usage
-----
    python pipeline.py                          # all models, all batches
    python pipeline.py --model Ume_Druma        # single model, all batches
    python pipeline.py --model Ume_Druma --batch I1KG3545  # single model + batch
    python pipeline.py --batch I1KG3545         # all models, one batch
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from eval_config import (
    BENCHMARK_DIR,
    LINE_DETECTION_MODEL,
    LINE_DETECTION_PATCH_SIZE,
    MODELS_CONFIG,
    NUM_WORKERS,
    OUTPUT_DIR,
)
from data_pipeline import get_all_image_paths, write_model_output
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ── Provider-specific helpers ────────────────────────────────────────────────

def _run_bdrc(model_name: str, settings: dict, images: list) -> list:
    """Initialise a BDRC OCRPipeline and run inference on all images."""
    from OCR.BDRC_models import initialize_pipeline, run_ocr_model

    pipeline = initialize_pipeline(
        model_dir=settings["model_dir"],
        line_model_path=LINE_DETECTION_MODEL,
        line_patch_size=LINE_DETECTION_PATCH_SIZE,
    )

    results = [None] * len(images)

    def _ocr_task(idx, image_name, batch_id, image_path):
        text = run_ocr_model(
            image_path=str(image_path),
            pipeline=pipeline,
            k_factor=settings["k_factor"],
            bbox_tolerance=settings["bbox_tolerance"],
            merge_lines=settings["merge_lines"],
            dewarp=settings["dewarp"],
            encoding=settings["encoding"],
        )
        return idx, image_name, batch_id, text

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_ocr_task, i, name, batch, path): i
            for i, (name, batch, path) in enumerate(images)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {model_name}",
            unit="img",
        ):
            idx, image_name, batch_id, text = future.result()
            results[idx] = (image_name, batch_id, text)

    return results


def _run_google_cloud_vision(model_name: str, settings: dict, images: list) -> list:
    """Run Google Cloud Vision TEXT_DETECTION on all images."""
    from OCR.google_cloud_vision import run_google_vision_ocr

    max_workers = settings.get("max_workers", NUM_WORKERS)
    results = [None] * len(images)

    def _ocr_task(idx, image_name, batch_id, image_path):
        text = run_google_vision_ocr(str(image_path))
        return idx, image_name, batch_id, text

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ocr_task, i, name, batch, path): i
            for i, (name, batch, path) in enumerate(images)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {model_name}",
            unit="img",
        ):
            idx, image_name, batch_id, text = future.result()
            results[idx] = (image_name, batch_id, text)

    return results


def _run_gemini(model_name: str, settings: dict, images: list) -> list:
    """Run Gemini vision model on all images."""
    from OCR.gemini import configure_gemini, run_gemini_ocr

    configure_gemini(api_key=settings["api_key"])
    gemini_model = settings.get("model_name", "gemini-2.0-flash")

    results = [None] * len(images)

    def _ocr_task(idx, image_name, batch_id, image_path):
        text = run_gemini_ocr(str(image_path), model_name=gemini_model)
        return idx, image_name, batch_id, text

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_ocr_task, i, name, batch, path): i
            for i, (name, batch, path) in enumerate(images)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {model_name}",
            unit="img",
        ):
            idx, image_name, batch_id, text = future.result()
            results[idx] = (image_name, batch_id, text)

    return results


def _run_deepseek(model_name: str, settings: dict, images: list) -> list:
    """Run DeepSeek vision model on all images."""
    from OCR.deepseek import run_deepseek_ocr

    api_key = settings["api_key"]
    base_url = settings.get("base_url", "https://api.deepseek.com")
    ds_model = settings.get("model_name", "deepseek-chat")

    results = [None] * len(images)

    def _ocr_task(idx, image_name, batch_id, image_path):
        text = run_deepseek_ocr(
            str(image_path),
            api_key=api_key,
            base_url=base_url,
            model=ds_model,
        )
        return idx, image_name, batch_id, text

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_ocr_task, i, name, batch, path): i
            for i, (name, batch, path) in enumerate(images)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {model_name}",
            unit="img",
        ):
            idx, image_name, batch_id, text = future.result()
            results[idx] = (image_name, batch_id, text)

    return results


# ── Provider dispatch table ──────────────────────────────────────────────────
_PROVIDERS = {
    "bdrc": _run_bdrc,
    "google_cloud_vision": _run_google_cloud_vision,
    "gemini": _run_gemini,
    "deepseek": _run_deepseek,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OCR models over benchmark images and save inference CSVs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODELS_CONFIG.keys()),
        help="Run a single model by name. "
             "If omitted, all configured models are run.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Run only on images from the given batch_id. "
             "If omitted, all batches are processed.",
    )
    return parser.parse_args()


def main():
    """Run configured OCR model(s) over benchmark images."""
    args = parse_args()

    # ── Resolve which models to run ──────────────────────────────────────
    if args.model:
        models_to_run = {args.model: MODELS_CONFIG[args.model]}
    else:
        models_to_run = MODELS_CONFIG

    # ── Collect benchmark images (optionally filtered by batch) ──────────
    images = get_all_image_paths(BENCHMARK_DIR, batch_id=args.batch)
    if not images:
        msg = f"No benchmark images found in {BENCHMARK_DIR}"
        if args.batch:
            msg += f" for batch '{args.batch}'"
        logger.error(msg)
        sys.exit(1)

    batch_info = f"batch '{args.batch}'" if args.batch else "all batches"
    logger.info(
        "Found %d benchmark images (%s) in %s",
        len(images), batch_info, BENCHMARK_DIR / "images",
    )

    # ── Iterate over each selected model ─────────────────────────────────
    for model_name, settings in models_to_run.items():
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("=" * 60)

        provider = settings.get("provider", "bdrc")
        runner = _PROVIDERS.get(provider)
        if runner is None:
            logger.error(
                "Unknown provider '%s' for model %s — skipping. "
                "Supported providers: %s",
                provider, model_name, ", ".join(_PROVIDERS),
            )
            continue

        logger.info("Provider: %s", provider)

        try:
            results = runner(model_name, settings, images)
        except Exception:
            logger.exception("Failed to run model %s — skipping.", model_name)
            continue

        # Write to CSV
        csv_path = write_model_output(model_name, results, OUTPUT_DIR)
        logger.info("Wrote %d rows to %s", len(results), csv_path)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
