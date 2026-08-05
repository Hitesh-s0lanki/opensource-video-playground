"""Wan 2.1 T2V-1.3B (Alibaba) - text-to-video.

Two Wan-specific details matter and neither is optional:

* the VAE must stay in fp32 - it produces NaNs in half precision;
* the scheduler needs flow-matching prediction with a resolution-dependent
  shift (3.0 at 480p-class resolutions, 5.0 for 720p+).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from .base import BackendUnavailable, GenerationRequest, ProgressCallback, VideoBackend

log = logging.getLogger(__name__)

#: Above this height the model wants the larger flow shift.
HIGH_RES_HEIGHT = 600
FLOW_SHIFT_LOW = 3.0
FLOW_SHIFT_HIGH = 5.0


class WanBackend(VideoBackend):
    def _build(self) -> Any:
        try:
            from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "Wan 2.1 needs diffusers >= 0.33. Run: uv pip install -U diffusers"
            ) from exc

        vae = AutoencoderKLWan.from_pretrained(
            self.spec.repo, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(self.spec.repo, vae=vae, torch_dtype=self.dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=FLOW_SHIFT_LOW,
        )
        return pipe

    def _set_flow_shift(self, height: int) -> None:
        shift = FLOW_SHIFT_HIGH if height >= HIGH_RES_HEIGHT else FLOW_SHIFT_LOW
        current = getattr(self.pipe.scheduler.config, "flow_shift", None)
        if current == shift:
            return
        from diffusers import UniPCMultistepScheduler

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
        log.info("Wan flow_shift set to %.1f for height %d", shift, height)

    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        self._set_flow_shift(request.height)

        output = self._call_pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            output_type="pil",
            callback_on_step_end=self._step_callback(progress, request.steps),
        )
        return output.frames[0]
