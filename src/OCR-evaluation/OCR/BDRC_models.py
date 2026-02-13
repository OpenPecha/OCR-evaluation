"""
BDRC OCR model integration.

Provides helpers to initialise a BDRC OCRPipeline and run inference on a
single image.  The tibetan-ocr-app directory is inserted into sys.path so
that its internal imports (Config, BDRC.*) resolve correctly.
"""

import sys
import os
import logging
from pathlib import Path

import cv2

# ── Make the tibetan-ocr-app importable ──────────────────────────────────────
# We insert at position 0 so that the BDRC package's own `import Config` finds
# the correct module (vendor/tibetan-ocr-app/Config.py) rather than any other
# `config` on the path.
from eval_config import TIBETAN_OCR_APP_DIR

_app_dir = str(TIBETAN_OCR_APP_DIR)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Now we can import from the BDRC package
from BDRC.Utils import import_local_model, get_platform  # noqa: E402
from BDRC.Data import Encoding, LineDetectionConfig       # noqa: E402
from BDRC.Inference import OCRPipeline                    # noqa: E402

logger = logging.getLogger(__name__)


def initialize_pipeline(
    model_dir: str,
    line_model_path: str,
    line_patch_size: int = 512,
) -> OCRPipeline:
    """
    Load an OCR model and build a ready-to-use OCRPipeline.

    Parameters
    ----------
    model_dir : str
        Path to the OCR model directory (must contain ``model_config.json``).
    line_model_path : str
        Path to the line-detection ONNX file
        (e.g. ``vendor/tibetan-ocr-app/Models/Lines/PhotiLines.onnx``).
    line_patch_size : int
        Patch size for line detection (default 512).

    Returns
    -------
    OCRPipeline
        An initialised pipeline ready for ``run_ocr()``.
    """
    ocr_model = import_local_model(model_dir)
    if ocr_model is None:
        raise FileNotFoundError(
            f"Could not load OCR model from {model_dir}. "
            "Make sure model_config.json exists inside the directory."
        )

    line_config = LineDetectionConfig(
        model_file=line_model_path,
        patch_size=line_patch_size,
    )

    platform = get_platform()
    pipeline = OCRPipeline(platform, ocr_model.config, line_config)
    logger.info("Initialised OCR pipeline with model %s", Path(model_dir).name)
    return pipeline


def run_ocr_model(
    image_path: str,
    pipeline: OCRPipeline,
    *,
    k_factor: float = 2.5,
    bbox_tolerance: float = 4.0,
    merge_lines: bool = True,
    dewarp: bool = False,
    encoding: str = "unicode",
) -> str:
    """
    Run OCR on a single image and return the recognised text.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    pipeline : OCRPipeline
        An initialised OCR pipeline (from ``initialize_pipeline``).
    k_factor, bbox_tolerance, merge_lines, dewarp, encoding
        OCR settings forwarded to ``pipeline.run_ocr()``.

    Returns
    -------
    str
        The OCR text (lines joined by ``\\n``), or an empty string when
        OCR fails for this image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Failed to read image: %s", image_path)
        return ""

    target_encoding = Encoding.Unicode if encoding == "unicode" else Encoding.Wylie

    try:
        status, result = pipeline.run_ocr(
            image=img,
            k_factor=k_factor,
            bbox_tolerance=bbox_tolerance,
            merge_lines=merge_lines,
            use_tps=dewarp,
            target_encoding=target_encoding,
        )
    except Exception:
        logger.exception("OCR crashed for %s", image_path)
        return ""

    if status.name == "SUCCESS":
        _rot_mask, _lines, ocr_lines, _angle = result
        text = "\n".join(line.text for line in ocr_lines)
        return text
    else:
        logger.warning("OCR failed for %s: %s", image_path, result)
        return ""
