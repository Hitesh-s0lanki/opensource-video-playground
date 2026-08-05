"""The catalogue of supported open-weight video models.

Each :class:`ModelSpec` carries everything the UI needs to render sane controls
(valid resolutions, the frame-count arithmetic the architecture demands, step
and guidance ranges) plus everything the backend needs to build its pipeline.
Adding a model = adding a spec here + a backend class in ``vidgen.backends``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_NEGATIVE_PROMPT, WAN_NEGATIVE_PROMPT

# Capability flags used across the UI.
T2V = "t2v"  # prompt -> video
I2V = "i2v"  # reference image becomes the literal first frame
REF = "ref"  # reference image steers style/subject (IP-Adapter), not frame 0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    repo: str
    backend: str
    params: str
    license: str
    modes: tuple[str, ...]
    dtype: str  # "fp16" | "bf16" | "fp32"

    # --- resolution ---
    resolutions: tuple[tuple[int, int], ...]
    default_resolution: tuple[int, int]
    dim_multiple: int = 8
    lock_resolution: bool = False

    # --- frame count: valid values are frame_base + k * frame_step ---
    default_frames: int = 25
    frame_min: int = 9
    frame_max: int = 129
    frame_step: int = 1
    frame_base: int = 1
    lock_frames: bool = False

    # --- sampling ---
    default_fps: int = 16
    default_steps: int = 30
    steps_range: tuple[int, int] = (10, 60)
    default_guidance: float = 6.0
    guidance_range: tuple[float, float] = (1.0, 12.0)
    supports_negative_prompt: bool = True
    default_negative: str = DEFAULT_NEGATIVE_PROMPT

    # --- housekeeping ---
    approx_download_gb: float = 10.0
    recommended_vram_gb: float = 8.0
    notes: str = ""
    extra_repos: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #

    @property
    def supports_t2v(self) -> bool:
        return T2V in self.modes

    @property
    def supports_i2v(self) -> bool:
        return I2V in self.modes

    @property
    def supports_ref(self) -> bool:
        return REF in self.modes

    @property
    def uses_reference_images(self) -> bool:
        return self.supports_i2v or self.supports_ref

    def snap_frames(self, n: int) -> int:
        """Round a requested frame count to the nearest architecturally valid one."""
        if self.lock_frames:
            return self.default_frames
        n = max(self.frame_min, min(self.frame_max, int(n)))
        if self.frame_step <= 1:
            return n
        k = round((n - self.frame_base) / self.frame_step)
        snapped = self.frame_base + k * self.frame_step
        # Clamping can push us off-grid again; walk back onto it.
        while snapped < self.frame_min:
            snapped += self.frame_step
        while snapped > self.frame_max:
            snapped -= self.frame_step
        return snapped

    def snap_dimension(self, v: int) -> int:
        m = self.dim_multiple
        return max(m, int(round(v / m)) * m)

    def duration_seconds(self, frames: int, fps: int) -> float:
        return frames / max(fps, 1)


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #

LTX_VIDEO = ModelSpec(
    key="ltx-video",
    name="LTX-Video (Lightricks)",
    repo="Lightricks/LTX-Video",
    backend="ltx",
    params="2B DiT",
    license="LTXV Open Weights License (commercial use permitted below a revenue cap)",
    modes=(T2V, I2V),
    dtype="bf16",
    resolutions=((512, 320), (640, 384), (704, 480), (768, 512), (960, 544)),
    default_resolution=(512, 320),
    dim_multiple=32,
    default_frames=73,
    frame_min=9,
    frame_max=161,
    frame_step=8,
    frame_base=1,
    default_fps=24,
    default_steps=30,
    steps_range=(8, 50),
    default_guidance=3.0,
    guidance_range=(1.0, 8.0),
    approx_download_gb=14.0,
    recommended_vram_gb=6.0,
    notes=(
        "Fastest open video model per frame. Dimensions must be multiples of 32 and "
        "frame count must satisfy 8n+1. Ships a T5-XXL text encoder, so the download "
        "is large even though the transformer itself is small."
    ),
)

COGVIDEOX_2B = ModelSpec(
    key="cogvideox-2b",
    name="CogVideoX-2B (Zhipu / THUDM)",
    repo="THUDM/CogVideoX-2b",
    backend="cogvideox",
    params="2B DiT",
    license="Apache-2.0",
    modes=(T2V,),
    dtype="fp16",
    resolutions=((720, 480),),
    default_resolution=(720, 480),
    lock_resolution=True,
    default_frames=49,
    frame_min=9,
    frame_max=49,
    frame_step=4,
    frame_base=1,
    default_fps=8,
    default_steps=40,
    steps_range=(20, 60),
    default_guidance=6.0,
    guidance_range=(1.0, 10.0),
    approx_download_gb=13.0,
    recommended_vram_gb=5.0,
    notes=(
        "Text-to-video only - the image-conditioned variant of CogVideoX is 5B and is "
        "out of reach here. Resolution is fixed at 720x480 by the checkpoint. Has the "
        "best-documented low-VRAM path of any model in this list."
    ),
)

WAN_T2V_1_3B = ModelSpec(
    key="wan21-t2v-1.3b",
    name="Wan 2.1 T2V-1.3B (Alibaba)",
    repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    backend="wan",
    params="1.3B DiT",
    license="Apache-2.0",
    modes=(T2V,),
    dtype="bf16",
    resolutions=((832, 480), (640, 384), (480, 480), (480, 832)),
    default_resolution=(640, 384),
    dim_multiple=16,
    default_frames=81,
    frame_min=17,
    frame_max=81,
    frame_step=4,
    frame_base=1,
    default_fps=16,
    default_steps=25,
    steps_range=(15, 50),
    default_guidance=5.0,
    guidance_range=(1.0, 10.0),
    default_negative=WAN_NEGATIVE_PROMPT,
    approx_download_gb=15.0,
    recommended_vram_gb=8.0,
    notes=(
        "Best prompt adherence and motion quality per parameter in this list. The VAE "
        "is kept in fp32 (it is numerically unstable in half precision), which costs a "
        "little extra RAM. Image-to-video for Wan 2.1 only exists at 14B."
    ),
)

SVD_XT = ModelSpec(
    key="svd-xt",
    name="Stable Video Diffusion XT (Stability AI)",
    repo="stabilityai/stable-video-diffusion-img2vid-xt",
    backend="svd",
    params="1.5B U-Net",
    license="Stability AI Community License (non-commercial above $1M revenue)",
    modes=(I2V,),
    dtype="fp16",
    # SVD's VAE downsamples by 8 and its U-Net by a further 8, so both
    # dimensions must be divisible by 64 - hence 320 rather than a true-16:9 288.
    resolutions=((448, 256), (512, 320), (640, 384), (768, 448), (1024, 576)),
    default_resolution=(512, 320),
    dim_multiple=64,
    default_frames=25,
    frame_min=25,
    frame_max=25,
    lock_frames=True,
    default_fps=7,
    default_steps=25,
    steps_range=(10, 40),
    default_guidance=3.0,
    guidance_range=(1.0, 8.0),
    supports_negative_prompt=False,
    approx_download_gb=9.5,
    recommended_vram_gb=6.0,
    notes=(
        "Pure image-to-video: it animates the reference image and ignores the text "
        "prompt entirely. Guidance here is the max end of a ramp applied across frames. "
        "A reference image is required."
    ),
)

ANIMATEDIFF_SD15 = ModelSpec(
    key="animatediff-sd15",
    name="AnimateDiff + SD 1.5",
    repo="emilianJR/epiCRealism",
    backend="animatediff",
    params="0.9B U-Net + 0.4B motion adapter",
    license="Apache-2.0 motion adapter / CreativeML OpenRAIL-M base",
    modes=(T2V, REF),
    dtype="fp16",
    resolutions=((512, 512), (512, 320), (640, 384), (768, 512)),
    default_resolution=(512, 512),
    dim_multiple=8,
    default_frames=16,
    frame_min=8,
    frame_max=32,
    frame_step=1,
    default_fps=8,
    default_steps=25,
    steps_range=(15, 40),
    default_guidance=7.5,
    guidance_range=(1.0, 15.0),
    approx_download_gb=9.5,
    recommended_vram_gb=4.0,
    notes=(
        "The only model here that runs comfortably on 4 GB. Reference images act as "
        "style/subject guidance through IP-Adapter rather than becoming frame 0. Lower "
        "fidelity than the DiT models, but the SD 1.5 ecosystem of LoRAs and checkpoints "
        "is enormous. The download is three parts: the SD 1.5 base, the motion adapter, "
        "and IP-Adapter with its CLIP image encoder."
    ),
    extra_repos=(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        "h94/IP-Adapter",
    ),
)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        LTX_VIDEO,
        ANIMATEDIFF_SD15,
        COGVIDEOX_2B,
        WAN_T2V_1_3B,
        SVD_XT,
    )
}


def get(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(
            f"Unknown model {key!r}. Available: {', '.join(sorted(MODELS))}"
        ) from None


def all_specs() -> list[ModelSpec]:
    """Catalogue order: lightest / most likely to succeed first."""
    return list(MODELS.values())
