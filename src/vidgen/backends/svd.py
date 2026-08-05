"""Stable Video Diffusion XT (Stability AI) - image-to-video only.

SVD has no text encoder: the reference image is the entire conditioning signal
and the prompt is ignored. Its two expressive controls are ``motion_bucket_id``
(how much movement, 1-255) and ``noise_aug_strength`` (how far it may drift from
the reference).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from .base import BackendUnavailable, GenerationRequest, ProgressCallback, VideoBackend

log = logging.getLogger(__name__)

DEFAULT_MOTION_BUCKET_ID = 127
DEFAULT_NOISE_AUG_STRENGTH = 0.02
#: SVD's conditioning fps is clamped by the model to this range.
FPS_RANGE = (5, 30)


class SVDBackend(VideoBackend):
    def _build(self) -> Any:
        try:
            from diffusers import StableVideoDiffusionPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "Stable Video Diffusion needs diffusers >= 0.24."
            ) from exc

        kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        # The fp16 variant halves the download, but only exists for half precision.
        if self.dtype == torch.float16:
            kwargs["variant"] = "fp16"

        try:
            return StableVideoDiffusionPipeline.from_pretrained(self.spec.repo, **kwargs)
        except Exception as exc:
            if "variant" not in kwargs:
                raise
            log.warning("fp16 variant unavailable (%s); retrying with default weights", exc)
            kwargs.pop("variant")
            return StableVideoDiffusionPipeline.from_pretrained(self.spec.repo, **kwargs)

    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        if not request.has_reference:
            raise ValueError("Stable Video Diffusion requires a reference image.")

        extra = request.extra
        motion_bucket_id = int(extra.get("motion_bucket_id", DEFAULT_MOTION_BUCKET_ID))
        noise_aug = float(extra.get("noise_aug_strength", DEFAULT_NOISE_AUG_STRENGTH))
        cond_fps = max(FPS_RANGE[0], min(FPS_RANGE[1], request.fps))

        # decode_chunk_size trades decode speed for peak VRAM; 1 is the smallest
        # possible working set and the only safe value on a 4 GB card.
        decode_chunk_size = 1 if self.hw.profile in ("low", "cpu") else 4

        output = self._call_pipe(
            image=request.reference_images[0],
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            num_inference_steps=request.steps,
            min_guidance_scale=1.0,
            max_guidance_scale=request.guidance_scale,
            fps=cond_fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
            output_type="pil",
            callback_on_step_end=self._step_callback(progress, request.steps),
        )
        return output.frames[0]
