# =============================================================================
# module_ingest/ocr.py - owner: Person 1 (Ingestion & OCR)
# Receives : an image path (.png .jpg .jpeg) from parser.parse_file.
# Hands off: plain extracted text (str) back to parser.parse_file.
# Engine   : pytesseract + the LOCAL Tesseract binary only.
# =============================================================================
"""Local OCR for images of notes (whiteboards, screenshots, scans)."""

from __future__ import annotations

import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)


def image_to_text(path: Path) -> str:
    """Return the text found in the image at ``path`` using local Tesseract.

    Inputs:
        path: image file (.png .jpg .jpeg).
    Output:
        Extracted text, whitespace-trimmed. "" if no text was found.
    Edge cases:
        - Tesseract binary missing -> log a clear hint ("install Tesseract or set
          TESSERACT_CMD") and return "" (parse_file then returns None).
        - Corrupt / unreadable image -> log and return "".
        - Converts to grayscale ("L") before OCR for better results.
    """
    import pytesseract
    from PIL import Image, UnidentifiedImageError

    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            text = pytesseract.image_to_string(gray)
            return text.strip()
    except pytesseract.TesseractNotFoundError:
        log.error(
            "Tesseract binary not found for %s. Install Tesseract or set TESSERACT_CMD.",
            path,
        )
        return ""
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        log.warning("Unreadable or corrupt image %s: %s", path, exc)
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("Unexpected OCR error for %s: %s", path, exc)
        return ""