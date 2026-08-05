"""LTX-Video (Lightricks) - text-to-video and image-to-video.

The two pipelines share every weight, so the image-conditioned variant is built
from the text pipeline's components on first use. That costs no extra VRAM and
no extra download.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
from PIL import Image

from .base import BackendUnavailable, GenerationRequest, ProgressCallback, VideoBackend

log = logging.getLogger(__name__)

# LTX decodes from a heavily compressed latent space; these two values control
# the denoise level the VAE decoder assumes and are the reference defaults.
DECODE_TIMESTEP = 0.03
DECODE_NOISE_SCALE = 0.025


class LTXBackend(VideoBackend):
    def __init__(self, spec, hw) -> None:
        super().__init__(spec, hw)
        self._i2v_pipe: Any = None

    def _build(self) -> Any:
        try:
            from diffusers import LTXPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "LTX-Video needs diffusers >= 0.32. Run: uv pip install -U diffusers"
            ) from exc

        return LTXPipeline.from_pretrained(self.spec.repo, torch_dtype=self.dtype)

    def unload(self) -> None:
        self._i2v_pipe = None
        super().unload()

    # ------------------------------------------------------------------ #

    def _image_pipeline(self) -> Any:
        if self._i2v_pipe is None:
            from diffusers import LTXImageToVideoPipeline

            # Sharing ``components`` reuses the already-offloaded modules; the
            # accelerate hooks live on the modules, not on the pipeline object.
            self._i2v_pipe = LTXImageToVideoPipeline(**self.pipe.components)
            try:
                self._i2v_pipe.set_progress_bar_config(disable=True)
            except AttributeError:
                pass
        return self._i2v_pipe

    @contextmanager
    def _using(self, pipeline: Any) -> Iterator[None]:
        """Temporarily make ``pipeline`` the active one for the helper methods."""
        original = self.pipe
        self.pipe = pipeline
        try:
            yield
        finally:
            self.pipe = original

    # ------------------------------------------------------------------ #

    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        pipeline = self._image_pipeline() if request.has_reference else self.pipe

        with self._using(pipeline):
            kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or None,
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "decode_timestep": DECODE_TIMESTEP,
                "decode_noise_scale": DECODE_NOISE_SCALE,
                "generator": generator,
                "output_type": "pil",
                "callback_on_step_end": self._step_callback(progress, request.steps),
            }
            if request.has_reference:
                kwargs["image"] = request.reference_images[0]
                if len(request.reference_images) > 1:
                    log.info("LTX conditions on a single frame; using the first reference image.")

            output = self._call_pipe(**kwargs)

        return output.frames[0]
