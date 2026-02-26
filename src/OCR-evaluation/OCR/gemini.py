"""
Google Gemini vision OCR integration.

Uses the ``google-genai`` SDK to send benchmark images to a Gemini model
and extract Tibetan text.

Prerequisites
-------------
1. ``pip install google-genai``
2. Set the ``GEMINI_API_KEY`` environment variable, **or** pass the key
   directly via the ``api_key`` parameter in :func:`configure_gemini`.

Usage
-----
    from OCR.gemini import configure_gemini, run_gemini_ocr

    configure_gemini(api_key="YOUR_KEY")
    text = run_gemini_ocr("path/to/image.tif", model_name="gemini-2.5-flash")
"""

import io
import logging
import mimetypes
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning(
        "google-genai is not installed. Install it with: pip install google-genai"
    )

_client = None

# Formats the Gemini API accepts natively (no conversion needed).
_GEMINI_SUPPORTED = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}

# Prompt used for all Gemini OCR calls.
_OCR_PROMPT = (
    "Extract all the Tibetan text from this image exactly as it appears. "
    "Return only the extracted text with no additional commentary or formatting."
)


def configure_gemini(api_key: str | None = None) -> None:
    """Initialise (or re-initialise) the Gemini client.

    Parameters
    ----------
    api_key : str or None
        Google AI API key.  If *None*, the SDK will fall back to the
        ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` environment variable.
    """
    global _client

    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install google-genai"
        )

    _client = genai.Client(api_key=api_key) if api_key else genai.Client()
    logger.info("Gemini client configured.")


def _get_client() -> "genai.Client":
    """Return the current Gemini client, creating one if needed."""
    global _client
    if _client is None:
        configure_gemini()
    return _client


def _mime_type_for(path: Path) -> str:
    """Guess MIME type from the file extension."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def run_gemini_ocr(image_path: str, *, model_name: str = "gemini-2.0-flash") -> str:
    """
    Run a Gemini vision model on a single image and return the detected text.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model_name : str
        Gemini model identifier (e.g. ``"gemini-2.0-flash"``).

    Returns
    -------
    str
        The detected text, or an empty string on failure.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file does not exist: %s", image_path)
        return ""

    client = _get_client()

    if path.suffix.lower() not in _GEMINI_SUPPORTED:
        logger.debug("Converting %s to PNG for Gemini compatibility", path.name)
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        mime = "image/png"
    else:
        mime = _mime_type_for(path)
        with open(path, "rb") as f:
            image_bytes = f.read()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                _OCR_PROMPT,
            ],
        )
    except Exception:
        logger.exception(
            "Gemini API call failed for %s (model=%s)", image_path, model_name
        )
        return ""

    if not response or not response.text:
        logger.info("No text returned by Gemini for %s", image_path)
        return ""

    return response.text.strip()


def _load_image_bytes(image_path: str) -> tuple[bytes, str]:
    """Load image from *image_path*, converting to PNG if needed.

    Returns ``(image_bytes, mime_type)``.
    """
    path = Path(image_path)
    if path.suffix.lower() not in _GEMINI_SUPPORTED:
        logger.debug("Converting %s to PNG for Gemini compatibility", path.name)
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    mime = _mime_type_for(path)
    with open(path, "rb") as f:
        return f.read(), mime


def run_gemini_ocr_structured(
    image_path: str, *, model_name: str = "gemini-3-flash-preview"
) -> tuple[str, str]:
    """Run structured Gemini OCR using the prompt schema from *gemini_prompt*.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model_name : str
        Gemini model identifier.

    Returns
    -------
    tuple[str, str]
        ``(raw_json_str, extracted_text)`` where *raw_json_str* is the full
        JSON response and *extracted_text* is the ``raw_transcription`` lines
        joined by newlines.
    """
    from OCR.gemini_prompt import (
        OCR_USER_PROMPT,
        TibetanManuscript,
        get_ocr_config,
    )

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file does not exist: %s", image_path)
        return "", ""

    client = _get_client()
    image_bytes, mime = _load_image_bytes(image_path)
    config = get_ocr_config(model_name)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                OCR_USER_PROMPT,
            ],
            config=config,
        )
    except Exception:
        logger.exception(
            "Gemini structured API call failed for %s (model=%s)",
            image_path, model_name,
        )
        return "", ""

    if not response or not response.text:
        logger.info("No text returned by Gemini for %s", image_path)
        return "", ""

    raw_json = response.text.strip()

    try:
        manuscript = TibetanManuscript.model_validate_json(raw_json)
        text = "\n".join(line.raw_transcription for line in manuscript.lines)
    except Exception:
        logger.warning(
            "Failed to parse structured JSON for %s — using raw text",
            image_path,
        )
        text = raw_json

    return raw_json, text
