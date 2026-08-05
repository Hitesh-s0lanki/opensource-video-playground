"""Answer "is this model already on disk?" without touching the network.

Used to keep the UI honest: a first run really does pull 7-15 GB, but saying so
on every render - when the weights are already cached and the wait is just a
disk read - trains people to ignore the message.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..registry import ModelSpec

__all__ = ["repo_cache_dir", "repo_is_cached", "cached_size_gb", "spec_is_cached"]


def _hub_root() -> Path:
    """The Hugging Face hub cache directory, honouring HF_HOME."""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def repo_cache_dir(repo: str) -> Path:
    """Where huggingface_hub stores ``org/name``."""
    return _hub_root() / ("models--" + repo.replace("/", "--"))


def repo_is_cached(repo: str) -> bool:
    """True when at least one complete snapshot of ``repo`` is on disk.

    Deliberately conservative: an interrupted download leaves ``*.incomplete``
    blobs behind, and a directory holding only those should not count as cached.
    """
    snapshots = repo_cache_dir(repo) / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False

    blobs = repo_cache_dir(repo) / "blobs"
    if blobs.is_dir():
        files = list(blobs.iterdir())
        if files and all(f.name.endswith(".incomplete") for f in files):
            return False
    return True


def spec_is_cached(spec: ModelSpec) -> bool:
    """True only when the model *and* every auxiliary repo it needs are present."""
    return all(repo_is_cached(repo) for repo in (spec.repo, *spec.extra_repos))


def cached_size_gb(spec: ModelSpec) -> float:
    """How much disk this model currently occupies, in GB."""
    total = 0
    for repo in (spec.repo, *spec.extra_repos):
        directory = repo_cache_dir(repo)
        if not directory.is_dir():
            continue
        total += sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / 1024**3
