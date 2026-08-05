"""Backend registry: maps a :class:`ModelSpec` to its implementation."""

from __future__ import annotations

from typing import Type

from ..hardware import Hardware
from ..registry import ModelSpec
from .animatediff import AnimateDiffBackend
from .base import (
    BackendUnavailable,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
    VideoBackend,
)
from .cogvideox import CogVideoXBackend
from .ltx import LTXBackend
from .svd import SVDBackend
from .wan import WanBackend

BACKENDS: dict[str, Type[VideoBackend]] = {
    "ltx": LTXBackend,
    "cogvideox": CogVideoXBackend,
    "wan": WanBackend,
    "svd": SVDBackend,
    "animatediff": AnimateDiffBackend,
}


def create(spec: ModelSpec, hw: Hardware) -> VideoBackend:
    try:
        backend_cls = BACKENDS[spec.backend]
    except KeyError:
        raise BackendUnavailable(
            f"No backend implementation registered for {spec.backend!r} "
            f"(model {spec.key!r})."
        ) from None
    return backend_cls(spec, hw)


__all__ = [
    "BACKENDS",
    "BackendUnavailable",
    "GenerationRequest",
    "GenerationResult",
    "ProgressCallback",
    "VideoBackend",
    "create",
]
