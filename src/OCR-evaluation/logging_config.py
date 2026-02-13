"""
Centralised logging configuration for the OCR evaluation project.

Call ``setup_logging()`` once at application start-up (in ``pipeline.py`` or
``CER_evaluation.py``).  Every module that does ``logging.getLogger(__name__)``
will automatically inherit the configured handlers.

Logs are written to:
  - **console** (stderr) at INFO level with colour-coded level names
  - **file** (``logs/ocr_evaluation.log``) at DEBUG level for full traceability
"""

import logging
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "ocr_evaluation.log"

# ── Formats ───────────────────────────────────────────────────────────────────
_CONSOLE_FMT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
_FILE_FMT = "%(asctime)s  %(levelname)-8s  [%(name)s:%(lineno)d]  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: Path | None = None,
) -> None:
    """
    Configure the root logger with a console handler and a rotating file handler.

    Parameters
    ----------
    console_level : int
        Minimum level for messages printed to the terminal (default INFO).
    file_level : int
        Minimum level for messages written to the log file (default DEBUG).
    log_file : Path | None
        Override the default log file path.
    """
    log_path = log_file or LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # ── File handler ──────────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(file_handler)

    root.info("Logging initialised  —  console=%s  file=%s",
              logging.getLevelName(console_level), log_path)
