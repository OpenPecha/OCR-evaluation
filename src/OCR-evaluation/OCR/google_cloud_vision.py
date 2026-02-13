"""
Google Cloud Vision OCR integration.

Provides a thin wrapper around the Cloud Vision TEXT_DETECTION API so that it
can be used interchangeably with the BDRC pipeline in the evaluation harness.

Prerequisites
-------------
1. ``pip install google-cloud-vision``
2. Set the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to point at
   a service-account key JSON file, **or** run on a GCP instance that already
   has default credentials configured.

Usage
-----
    from OCR.google_cloud_vision import run_google_vision_ocr

    text = run_google_vision_ocr("path/to/image.jpg")
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision  # type: ignore[import-untyped]

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False
    logger.warning(
        "google-cloud-vision is not installed.  "
        "Install it with: pip install google-cloud-vision"
    )


def _get_client() -> "vision.ImageAnnotatorClient":
    """Return a cached Vision API client (created on first call)."""
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "google-cloud-vision is not installed. "
            "Run: pip install google-cloud-vision"
        )
    if not hasattr(_get_client, "_client"):
        _get_client._client = vision.ImageAnnotatorClient()  # type: ignore[attr-defined]
    return _get_client._client  # type: ignore[attr-defined]


def run_google_vision_ocr(image_path: str) -> str:
    """
    Run Google Cloud Vision TEXT_DETECTION on a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    str
        The full detected text (``full_text_annotation.text``), or an empty
        string when no text is found or an error occurs.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file does not exist: %s", image_path)
        return ""

    client = _get_client()

    with open(path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)  # type: ignore[union-attr]

    try:
        response = client.text_detection(image=image)
    except Exception:
        logger.exception("Google Cloud Vision API call failed for %s", image_path)
        return ""

    if response.error.message:
        logger.error(
            "Vision API error for %s: %s", image_path, response.error.message
        )
        return ""

    if not response.text_annotations:
        logger.info("No text detected in %s", image_path)
        return ""

    # The first annotation contains the full concatenated text
    full_text = response.text_annotations[0].description
    return full_text.strip()


def run_google_vision_ocr_batch(
    image_paths: list,
    *,
    max_workers: int = 4,
) -> dict:
    """
    Run Google Cloud Vision OCR on multiple images concurrently.

    Parameters
    ----------
    image_paths : list of str
        Paths to image files.
    max_workers : int
        Number of concurrent API calls (default 4).

    Returns
    -------
    dict
        Mapping ``{image_path: detected_text}``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_google_vision_ocr, p): p for p in image_paths
        }
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                results[img_path] = future.result()
            except Exception:
                logger.exception("OCR failed for %s", img_path)
                results[img_path] = ""
    return results
