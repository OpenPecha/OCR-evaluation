"""
Google Cloud Vision OCR integration.

Provides a thin wrapper around the Cloud Vision DOCUMENT_TEXT_DETECTION API
(with Tibetan language hint) so that it can be used interchangeably with the
BDRC pipeline in the evaluation harness.

Prerequisites
-------------
1. ``pip install google-cloud-vision``
2. Set the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to point at
   a service-account key JSON file, **or** run on a GCP instance that already
   has default credentials configured.

Usage
-----
    from OCR.google_cloud_vision import run_google_vision_ocr

    text, raw = run_google_vision_ocr("path/to/image.jpg")
"""

import io
import logging
from pathlib import Path
from typing import Tuple

from PIL import Image

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


# Formats the Cloud Vision API accepts natively (no conversion needed).
_VISION_SUPPORTED = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".tif", ".tiff", ".pdf",
}


def _load_image_bytes(path: Path) -> bytes:
    """Read image bytes, converting to PNG if the format is not natively supported."""
    if path.suffix.lower() in _VISION_SUPPORTED:
        with open(path, "rb") as f:
            return f.read()
    logger.debug("Converting %s to PNG for Vision API compatibility", path.name)
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_google_vision_ocr(image_path: str) -> Tuple[str, dict]:
    """
    Run Google Cloud Vision DOCUMENT_TEXT_DETECTION on a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple of (str, dict)
        A pair of (detected_text, raw_response_dict).
        ``detected_text`` is the full detected text
        (``full_text_annotation.text``), or an empty string when no text is
        found or an error occurs.
        ``raw_response_dict`` is the full API response serialized as a dict,
        or an empty dict on failure.
    """
    from google.protobuf.json_format import MessageToDict

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file does not exist: %s", image_path)
        return "", {}

    client = _get_client()
    content = _load_image_bytes(path)

    image = vision.Image(content=content)  # type: ignore[union-attr]
    image_context = vision.ImageContext(language_hints=["bo"])  # type: ignore[union-attr]

    try:
        response = client.document_text_detection(
            image=image,
            image_context=image_context,
        )
    except Exception:
        logger.exception("Google Cloud Vision API call failed for %s", image_path)
        return "", {}

    raw_dict = MessageToDict(response._pb)

    if response.error.message:
        logger.error(
            "Vision API error for %s: %s", image_path, response.error.message
        )
        return "", raw_dict

    annotation = response.full_text_annotation
    if not annotation or not annotation.text:
        logger.info("No text detected in %s", image_path)
        return "", raw_dict

    return annotation.text.strip(), raw_dict


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
        Mapping ``{image_path: (detected_text, raw_response_dict)}``.
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
                results[img_path] = ("", {})
    return results
