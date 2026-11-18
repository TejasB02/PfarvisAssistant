# =============================================================================
# scripts/prefetch_models.py - ONE-TIME ONLINE SETUP (shared; Person 2 maintains)
# Receives : nothing. Run it once, WITH internet, before the demo:
#                python scripts/prefetch_models.py
# Hands off: models/all-MiniLM-L6-v2/ on disk (config.EMBED_MODEL_PATH), used
#            offline by module_storage.vector_db; prints the Ollama pull command.
# RULE     : the ONLY file allowed to reach the internet. Nothing imports it.
#            It clears the offline variables for ITS OWN PROCESS only.
# =============================================================================
"""Download the embedding model into models/ and print the Ollama pull command."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (paths and model names only)

HUB_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def main() -> int:
    target = Path(config.EMBED_MODEL_PATH)
    force = "--force" in sys.argv

    if (target / "config.json").exists() and not force:
        print(f"Embedding model already present at {target} (use --force to re-download).")
    else:
        # config set the offline switches; lift them for this process only,
        # BEFORE sentence_transformers / huggingface_hub are imported.
        for var in OFFLINE_VARS:
            os.environ.pop(var, None)
        from sentence_transformers import SentenceTransformer

        print(f"Downloading {HUB_MODEL_ID} ...")
        model = SentenceTransformer(HUB_MODEL_ID, device="cpu")
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target))
        # Sanity check: load back from the local folder and embed one sentence.
        dim = len(SentenceTransformer(str(target), device="cpu").encode("hello"))
        print(f"Saved to {target} (embedding size {dim}).")

    print()
    print("Next, pull the local LLM (needs internet once):")
    print(f"    ollama pull {config.OLLAMA_MODEL}")
    print()
    if shutil.which("tesseract") or config.TESSERACT_CMD:
        print("Tesseract found: OCR for images will work.")
    else:
        print("Tesseract NOT found on PATH: install it, or set TESSERACT_CMD to tesseract.exe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
