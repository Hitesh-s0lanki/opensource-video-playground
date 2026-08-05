"""CogVideoX-2B (Zhipu / THUDM) - text-to-video.

This is the friendliest model here for a small card: with sequential CPU offload
plus VAE tiling and slicing the reference implementation reports a working set
under 4 GB. Resolution is fixed at 720x480 by the checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from .base import BackendUnavailable, GenerationRequest, ProgressCallback, VideoBackend

log = logging.getLogger(__name__)


class CogVideoXBackend(VideoBackend):
    def _build(self) -> Any:
        try:
            from diffusers import CogVideoXPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "CogVideoX needs diffusers >= 0.30. Run: uv pip install -U diffusers"
            ) from exc

        return CogVideoXPipeline.from_pretrained(self.spec.repo, torch_dtype=self.dtype)

    def _apply_memory_profile(self) -> None:
        super()._apply_memory_profile()
        # Cog's VAE is the peak-memory step by a wide margin, so slicing is
        # enabled on every profile here - not just the offloaded ones.
        self._enable_vae_saver("enable_slicing", "enable_vae_slicing")

    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        output = self._call_pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            num_frames=request.num_frames,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_videos_per_prompt=1,
            generator=generator,
            output_type="pil",
            callback_on_step_end=self._step_callback(progress, request.steps),
        )
        return output.frames[0]
