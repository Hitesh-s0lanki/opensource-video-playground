"""The backend contract shared by every model.

A backend owns exactly one diffusers pipeline. It knows how to build it, how to
fit it into the available VRAM, and how to translate a neutral
:class:`GenerationRequest` into whatever argument names that particular pipeline
expects. Everything above this layer is model-agnostic.
"""

from __future__ import annotations

import inspect
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import torch
from PIL import Image

from ..hardware import Hardware, free_memory, resolve_dtype
from ..registry import ModelSpec
from ..utils.images import prepare_reference

log = logging.getLogger(__name__)

#: ``(current_step, total_steps, message)`` - wired to the Streamlit progress bar.
ProgressCallback = Callable[[int, int, str], None]

SEED_MAX = 2**31 - 1


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependencies are missing or too old."""


@dataclass
class GenerationRequest:
    """A model-agnostic description of one render."""

    prompt: str = ""
    negative_prompt: str = ""
    reference_images: list[Image.Image] = field(default_factory=list)
    width: int = 512
    height: int = 320
    num_frames: int = 25
    fps: int = 16
    steps: int = 30
    guidance_scale: float = 5.0
    seed: int = -1
    reference_strength: float = 0.7
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_images)

    def resolved_seed(self) -> int:
        return random.randint(0, SEED_MAX) if self.seed is None or self.seed < 0 else int(self.seed)


@dataclass
class GenerationResult:
    frames: list[Image.Image]
    fps: int
    seed: int
    elapsed_seconds: float
    metadata: dict[str, Any]


class VideoBackend(ABC):
    """Base class: lifecycle, memory strategy, and safe pipeline invocation."""

    def __init__(self, spec: ModelSpec, hw: Hardware) -> None:
        self.spec = spec
        self.hw = hw
        self.pipe: Any = None
        self._load_seconds: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    @property
    def dtype(self) -> torch.dtype:
        return resolve_dtype(self.spec.dtype, self.hw)

    @abstractmethod
    def _build(self) -> Any:
        """Construct and return the diffusers pipeline (weights on CPU)."""

    def load(self, progress: ProgressCallback | None = None) -> None:
        if self.loaded:
            return
        if progress:
            from ..utils.cache import spec_is_cached

            progress(
                0,
                1,
                f"Loading {self.spec.name} from disk"
                if spec_is_cached(self.spec)
                else f"Downloading {self.spec.name} (~{self.spec.approx_download_gb:.0f} GB, one time)",
            )

        started = time.perf_counter()
        log.info("Building pipeline for %s (%s)", self.spec.key, self.spec.repo)
        self.pipe = self._build()
        self._apply_memory_profile()
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except AttributeError:
            pass
        self._load_seconds = time.perf_counter() - started
        log.info("Loaded %s in %.1fs", self.spec.key, self._load_seconds)

    def unload(self) -> None:
        if not self.loaded:
            return
        log.info("Unloading %s", self.spec.key)
        try:
            # Detach accelerate's offload hooks so submodules are actually freed.
            from accelerate.hooks import remove_hook_from_module

            for attr in ("transformer", "unet", "vae", "text_encoder", "image_encoder"):
                module = getattr(self.pipe, attr, None)
                if isinstance(module, torch.nn.Module):
                    remove_hook_from_module(module, recurse=True)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            log.debug("Hook removal skipped: %s", exc)

        self.pipe = None
        free_memory()

    # ------------------------------------------------------------------ #
    # Memory strategy
    # ------------------------------------------------------------------ #

    def _apply_memory_profile(self) -> None:
        """Choose where weights live, based on the detected profile.

        ``low``    sequential CPU offload - every block is moved to the GPU right
                   before it runs and evicted straight after. This is what makes a
                   4 GB card work at all; it is also why it is slow.
        ``medium`` model CPU offload - a whole submodule at a time.
        ``high``   everything resident on the GPU.
        """
        profile = self.hw.profile

        if profile == "cpu":
            self.pipe.to("cpu")
        elif profile == "low":
            self.pipe.enable_sequential_cpu_offload()
        elif profile == "medium":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        # VAE tiling/slicing is the difference between decoding 80 frames and
        # OOM-ing on the decode step, which is where most video OOMs happen.
        self._enable_vae_saver("enable_tiling", "enable_vae_tiling")

        if profile in ("low", "medium", "cpu"):
            self._try("enable_attention_slicing")
            self._enable_vae_saver("enable_slicing", "enable_vae_slicing")

        log.info("Memory profile %r applied to %s", profile, self.spec.key)

    def _enable_vae_saver(self, vae_method: str, pipe_method: str) -> bool:
        """Enable a VAE memory saver, preferring the VAE's own method.

        The pipeline-level ``enable_vae_*`` wrappers are deprecated as of
        diffusers 0.39 and slated for removal in 0.40, so go to the VAE directly
        and keep the wrapper only as a fallback for older/odd pipelines.
        """
        vae = getattr(self.pipe, "vae", None)
        method = getattr(vae, vae_method, None)
        if method is not None:
            try:
                method()
                return True
            except Exception as exc:  # pragma: no cover - optional optimisation
                log.debug("vae.%s() failed: %s", vae_method, exc)
        return self._try(pipe_method)

    def _try(self, method_name: str, *args: Any, **kwargs: Any) -> bool:
        """Call an optional pipeline method, returning whether it succeeded."""
        method = getattr(self.pipe, method_name, None)
        if method is None:
            return False
        try:
            method(*args, **kwargs)
            return True
        except Exception as exc:  # pragma: no cover - optional optimisations
            log.debug("%s() unavailable: %s", method_name, exc)
            return False

    # ------------------------------------------------------------------ #
    # Invocation helpers
    # ------------------------------------------------------------------ #

    def _call_pipe(self, **kwargs: Any) -> Any:
        """Invoke the pipeline, dropping kwargs this diffusers version rejects.

        Video pipelines churn their signatures release to release. Filtering
        here turns "TypeError: unexpected keyword argument" into a debug line.
        """
        signature = inspect.signature(self.pipe.__call__)
        params = signature.parameters
        accepts_var_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())

        if accepts_var_kwargs:
            accepted = kwargs
        else:
            accepted = {k: v for k, v in kwargs.items() if k in params}
            dropped = set(kwargs) - set(accepted)
            if dropped:
                log.debug("Dropping unsupported kwargs for %s: %s", self.spec.key, sorted(dropped))

        return self.pipe(**accepted)

    def _step_callback(self, progress: ProgressCallback | None, total: int) -> Any | None:
        """Adapt our progress callback to the diffusers ``callback_on_step_end`` protocol."""
        if progress is None:
            return None
        if "callback_on_step_end" not in inspect.signature(self.pipe.__call__).parameters:
            return None

        def _on_step_end(_pipe: Any, step: int, _timestep: Any, kwargs: dict) -> dict:
            progress(step + 1, total, f"Denoising step {step + 1}/{total}")
            return kwargs

        return _on_step_end

    def _generator(self, seed: int) -> torch.Generator:
        # A CPU generator keeps results reproducible regardless of which offload
        # strategy is active - device generators desync under sequential offload.
        return torch.Generator(device="cpu").manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Request normalisation
    # ------------------------------------------------------------------ #

    def normalize(self, request: GenerationRequest) -> GenerationRequest:
        """Snap a request onto this architecture's valid grid."""
        spec = self.spec

        if spec.lock_resolution:
            width, height = spec.default_resolution
        else:
            width = spec.snap_dimension(request.width)
            height = spec.snap_dimension(request.height)

        frames = spec.snap_frames(request.num_frames)

        refs = list(request.reference_images)
        if refs and spec.uses_reference_images:
            refs = [prepare_reference(img, width, height) for img in refs]
        elif refs and not spec.uses_reference_images:
            log.warning("%s ignores reference images; dropping %d", spec.key, len(refs))
            refs = []

        steps = int(min(max(request.steps, spec.steps_range[0]), spec.steps_range[1]))
        guidance = float(
            min(max(request.guidance_scale, spec.guidance_range[0]), spec.guidance_range[1])
        )
        negative = request.negative_prompt if spec.supports_negative_prompt else ""

        return replace(
            request,
            width=width,
            height=height,
            num_frames=frames,
            steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative,
            reference_images=refs,
        )

    def validate(self, request: GenerationRequest) -> None:
        """Raise ``ValueError`` if this request cannot possibly work."""
        spec = self.spec
        if spec.modes == ("i2v",) and not request.has_reference:
            raise ValueError(f"{spec.name} is image-to-video only - upload a reference image.")
        if not request.prompt.strip() and spec.supports_t2v and not request.has_reference:
            raise ValueError("Enter a prompt (or upload a reference image).")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _generate_frames(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        progress: ProgressCallback | None,
    ) -> Sequence[Image.Image]:
        """Run the pipeline and return PIL frames."""

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        self.validate(request)
        request = self.normalize(request)
        self.load(progress=progress)

        seed = request.resolved_seed()
        generator = self._generator(seed)

        log.info(
            "Generating with %s: %dx%d, %d frames, %d steps, cfg %.1f, seed %d",
            self.spec.key,
            request.width,
            request.height,
            request.num_frames,
            request.steps,
            request.guidance_scale,
            seed,
        )

        started = time.perf_counter()
        if progress:
            progress(0, request.steps, "Starting denoise loop")

        try:
            frames = self._generate_frames(request, generator, progress)
        except torch.cuda.OutOfMemoryError as exc:
            free_memory()
            raise RuntimeError(
                f"Ran out of VRAM on {self.hw.gpu_name}. Try a smaller resolution, "
                f"fewer frames, or a lighter model (AnimateDiff is the smallest here). "
                f"Original error: {exc}"
            ) from exc

        elapsed = time.perf_counter() - started
        frames = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames]

        log.info(
            "Generated %d frames in %.1fs (%.1fs/frame)",
            len(frames),
            elapsed,
            elapsed / max(len(frames), 1),
        )

        return GenerationResult(
            frames=frames,
            fps=request.fps,
            seed=seed,
            elapsed_seconds=elapsed,
            metadata=self._metadata(request, seed, elapsed),
        )

    def _metadata(self, request: GenerationRequest, seed: int, elapsed: float) -> dict[str, Any]:
        return {
            "model": self.spec.key,
            "model_name": self.spec.name,
            "repo": self.spec.repo,
            "license": self.spec.license,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "requested_frames": request.num_frames,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": seed,
            "reference_images": len(request.reference_images),
            "reference_strength": request.reference_strength if request.has_reference else None,
            "dtype": str(self.dtype).replace("torch.", ""),
            "memory_profile": self.hw.profile,
            "device": self.hw.gpu_name,
            "elapsed_seconds": round(elapsed, 1),
            "load_seconds": round(self._load_seconds, 1),
        }
