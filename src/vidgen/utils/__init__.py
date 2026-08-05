"""Shared helpers: image preparation, video export, logging setup."""

from .cache import cached_size_gb, repo_is_cached, spec_is_cached
from .images import load_image, prepare_reference, to_thumbnail
from .logging import setup_logging
from .video import (
    OutputRecord,
    list_outputs,
    read_metadata,
    save_video,
    slugify,
)

__all__ = [
    "cached_size_gb",
    "repo_is_cached",
    "spec_is_cached",
    "load_image",
    "prepare_reference",
    "to_thumbnail",
    "setup_logging",
    "OutputRecord",
    "list_outputs",
    "read_metadata",
    "save_video",
    "slugify",
]
