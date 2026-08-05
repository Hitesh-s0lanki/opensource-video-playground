"""AnimateDiff + SD 1.5, with IP-Adapter for reference-image guidance.

Here reference images do *not* become frame 0. They are encoded by CLIP and
injected into cross-attention, so the clip inherits the reference's subject and
style while the prompt still drives the motion. That is usually what people
actually want from a "reference image", and it is the only such path available
in this catalogue.

Implementation note: once IP-Adapter is loaded the pipeline always expects an
``ip_adapter_image``. For prompt-only renders we therefore pass a neutral grey
image at scale 0.0, which contributes nothing to the result.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from .base import BackendUnavailable, GenerationRequest, ProgressCallback, VideoBackend

log = logging.getLogger(__name__)

MOTION_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-2"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter_sd15.bin"

_NEUTRAL = Image.new("RGB", (224, 224), (127, 127, 127))


class AnimateDiffBackend(VideoBackend):
    def __init__(self, spec, hw) -> None:
        super().__init__(spec, hw)
        self._ip_adapter_ready = False

    def _build(self) -> Any:
        try:
            from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
        except ImportError as exc:
            raise BackendUnavailable(
                "AnimateDiff needs diffusers >= 0.25. Run: uv pip install -U diffusers"
            ) from exc

        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_REPO, torch_dtype=self.dtype)
        pipe = AnimateDiffPipeline.from_pretrained(
            self.spec.repo, motion_adapter=adapter, torch_dtype=self.dtype
        )

        # The motion modules were trained against this exact DDIM configuration;
        # the defaults shipped with most SD 1.5 checkpoints produce mush.
        pipe.scheduler = DDIMScheduler.from_pretrained(
            self.spec.repo,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        # Loaded before the offload hooks are installed, otherwise the extra
        # image encoder never gets wrapped and ends up stranded on the CPU.
        try:
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHTS
            )
            self._ip_adapter_ready = True
        except Exception as exc:
            log.warning(
                "IP-Adapter unavailable (%s). Reference images will be ignored "
                "for this model; text-to-video still works.",
                exc,
            )
            self._ip_adapter_ready = False

        return pipe

    def unload(self) -> None:
        self._ip_adapter_ready = False
        super().unload()

    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or None,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "output_type": "pil",
            "callback_on_step_end": self._step_callback(progress, request.steps),
        }

        if self._ip_adapter_ready:
            if request.has_reference:
                self.pipe.set_ip_adapter_scale(request.reference_strength)
                # SD 1.5's IP-Adapter accepts a list and averages the embeddings,
                # so multiple references blend into one style signal.
                images = request.reference_images
                kwargs["ip_adapter_image"] = images if len(images) > 1 else images[0]
            else:
                self.pipe.set_ip_adapter_scale(0.0)
                kwargs["ip_adapter_image"] = _NEUTRAL

        output = self._call_pipe(**kwargs)
        return output.frames[0]
