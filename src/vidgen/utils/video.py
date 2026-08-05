"""Video export and the on-disk output gallery.

Every render writes two files side by side: ``<stem>.mp4`` and ``<stem>.json``.
The JSON sidecar is the whole provenance record - model, seed, prompt, every
sampler setting - so a clip you like can be reproduced or nudged later.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..config import OUTPUT_DIR, VIDEO_SUFFIX

log = logging.getLogger(__name__)

__all__ = [
    "OutputRecord",
    "list_outputs",
    "read_metadata",
    "save_video",
    "slugify",
]

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe fragment of a prompt, for human-scannable filenames."""
    slug = _SLUG_STRIP.sub("-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "untitled"


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def save_video(
    frames: Sequence[Image.Image],
    fps: int,
    metadata: dict[str, Any],
    output_dir: Path | None = None,
    stem_hint: str = "",
) -> Path:
    """Encode ``frames`` to H.264 MP4 and write the metadata sidecar.

    Returns the path to the written video.
    """
    if not frames:
        raise ValueError("Cannot save a video with zero frames.")

    directory = Path(output_dir or OUTPUT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}_{slugify(stem_hint)}" if stem_hint else timestamp
    path = _unique_path(directory, stem, VIDEO_SUFFIX)

    _encode(frames, path, fps)

    record = dict(metadata)
    record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    record["fps"] = fps
    record["num_frames"] = len(frames)
    record["duration_seconds"] = round(len(frames) / max(fps, 1), 2)
    record["file"] = path.name

    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Wrote %s (%d frames @ %d fps)", path, len(frames), fps)
    return path


def _encode(frames: Sequence[Image.Image], path: Path, fps: int) -> None:
    """Write MP4 via diffusers' helper, falling back to imageio directly.

    ``export_to_video`` is the well-tested path but its signature has moved
    around between diffusers releases, so we keep a direct writer in reserve.
    """
    try:
        from diffusers.utils import export_to_video

        export_to_video(list(frames), str(path), fps=fps)
        return
    except Exception as exc:  # pragma: no cover - depends on installed version
        log.warning("export_to_video failed (%s); falling back to imageio", exc)

    import imageio.v2 as imageio
    import numpy as np

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # do not silently resize odd dimensions
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


@dataclass(frozen=True)
class OutputRecord:
    """One previously generated clip plus whatever metadata survived."""

    path: Path
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def created_at(self) -> str:
        return self.metadata.get("created_at") or datetime.fromtimestamp(
            self.path.stat().st_mtime
        ).isoformat(timespec="seconds")

    @property
    def prompt(self) -> str:
        return self.metadata.get("prompt", "")

    @property
    def model(self) -> str:
        return self.metadata.get("model_name") or self.metadata.get("model", "unknown")


def read_metadata(video_path: Path) -> dict[str, Any]:
    sidecar = video_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Unreadable metadata for %s: %s", video_path.name, exc)
        return {}


def list_outputs(output_dir: Path | None = None, limit: int | None = None) -> list[OutputRecord]:
    """Every rendered clip, newest first."""
    directory = Path(output_dir or OUTPUT_DIR)
    if not directory.exists():
        return []

    videos = sorted(
        directory.glob(f"*{VIDEO_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if limit is not None:
        videos = videos[:limit]

    return [OutputRecord(path=v, metadata=read_metadata(v)) for v in videos]
