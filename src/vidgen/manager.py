"""Model lifecycle management and the top-level render entry point.

Only one backend is ever resident. Switching models unloads the previous one
first - on a small card, holding two video pipelines in RAM is what turns a
slow render into a swap-thrashing one.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path
from typing import Any

from . import hardware
from .backends import GenerationRequest, GenerationResult, VideoBackend, create
from .hardware import Hardware
from .registry import ModelSpec, get as get_spec
from .utils.video import save_video

log = logging.getLogger(__name__)


class ModelManager:
    """Owns the single resident backend and serialises access to it."""

    def __init__(self, hw: Hardware | None = None) -> None:
        self.hw = hw or hardware.detect()
        self._backend: VideoBackend | None = None
        self._lock = threading.RLock()
        log.info("ModelManager on %s (profile=%s)", self.hw.summary(), self.hw.profile)

    # ------------------------------------------------------------------ #

    @property
    def active_key(self) -> str | None:
        return self._backend.spec.key if self._backend else None

    @property
    def active_spec(self) -> ModelSpec | None:
        return self._backend.spec if self._backend else None

    def with_profile(self, profile: str) -> None:
        """Override the auto-detected memory profile, reloading if necessary."""
        if profile == self.hw.profile:
            return
        log.info("Memory profile override: %s -> %s", self.hw.profile, profile)
        self.hw = dataclasses.replace(self.hw, profile=profile)
        self.unload()

    def get(self, model_key: str) -> VideoBackend:
        """Return a backend for ``model_key``, evicting any other resident model."""
        with self._lock:
            if self._backend is not None and self._backend.spec.key == model_key:
                return self._backend

            if self._backend is not None:
                log.info("Switching %s -> %s", self._backend.spec.key, model_key)
                self._backend.unload()
                self._backend = None
                hardware.free_memory()

            self._backend = create(get_spec(model_key), self.hw)
            return self._backend

    def unload(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.unload()
                self._backend = None
            hardware.free_memory()

    # ------------------------------------------------------------------ #

    def render(
        self,
        model_key: str,
        request: GenerationRequest,
        progress: Any | None = None,
        output_dir: Path | None = None,
        save: bool = True,
    ) -> tuple[GenerationResult, Path | None]:
        """Generate a clip and (by default) write it to disk with its metadata."""
        with self._lock:
            backend = self.get(model_key)
            result = backend.generate(request, progress=progress)

        path: Path | None = None
        if save:
            if progress:
                # 1/1 rather than request.steps: the request here is the raw one,
                # whose step count may have been clamped during normalisation.
                progress(1, 1, "Encoding video")
            path = save_video(
                frames=result.frames,
                fps=result.fps,
                metadata=result.metadata,
                output_dir=output_dir,
                stem_hint=request.prompt or backend.spec.key,
            )
        return result, path


_GLOBAL: ModelManager | None = None


def get_manager() -> ModelManager:
    """Process-wide singleton, used by the CLI. Streamlit uses its own cache."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ModelManager()
    return _GLOBAL
