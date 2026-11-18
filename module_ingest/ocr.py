# =============================================================================
# module_ingest/ocr.py - owner: Person 1 (Ingestion & OCR)
# Receives : an image path (.png .jpg .jpeg) from parser.parse_file.
# Hands off: plain extracted text (str) back to parser.parse_file.
# Engine   : pytesseract + the LOCAL Tesseract binary only.
# STATUS   : STUB - returns a mock string.
# =============================================================================
"""Local OCR for images of notes (whiteboards, screenshots, scans)."""

from __future__ import annotations

import logging
from pathlib import Path

import config  # noqa: F401  (config.TESSERACT_CMD is used by the real implementation)

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
        - Consider converting to grayscale ("L") before OCR for better results.
    """
    # TODO(person 1): real OCR, imports inside the function:
    #   import pytesseract
    #   from PIL import Image
    #   if config.TESSERACT_CMD:
    #       pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    #   with Image.open(path) as img:
    #       return pytesseract.image_to_string(img.convert("L")).strip()
    log.debug("Mock OCR for %s", path)
    return f"(mock OCR text from {Path(path).name})"
