"""Launch Pfarvis Assistant with the local notes pipeline."""

import config  # noqa: F401 - configure offline mode before importing the app

import logging

from module_ui.rag_app import run

if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()
