"""Logging setup shared by the CLI and the Streamlit app."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once, quieting the noisier third-party loggers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("VIDGEN_LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # httpcore in particular logs every TLS handshake and header at DEBUG, which
    # buries our own output the moment anyone passes -v during a model download.
    for noisy in (
        "urllib3",
        "filelock",
        "httpx",
        "httpcore",
        "PIL",
        "matplotlib",
        "huggingface_hub",
        "hf_xet",
        "fsspec",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # diffusers/transformers emit long progress-bar chatter we replace with our own.
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _CONFIGURED = True
