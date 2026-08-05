"""GPU/RAM detection and the memory *profile* that drives offload strategy.

The profile is the single most important knob in this project. On a 4 GB card
nothing loads unless weights are streamed block-by-block from system RAM, and
on a 24 GB card doing that would waste most of the run in PCIe transfers.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Literal

import psutil
import torch

log = logging.getLogger(__name__)

Profile = Literal["cpu", "low", "medium", "high"]

#: VRAM (GiB) below which we stream every block from system RAM.
LOW_VRAM_CEILING = 6.5
#: VRAM (GiB) below which we keep whole submodules on CPU between calls.
MEDIUM_VRAM_CEILING = 13.0

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "cpu": "No CUDA device. Generation runs on CPU and will take hours - not recommended.",
    "low": "Sequential CPU offload: weights stream per-block from RAM. Fits tiny VRAM, slowest.",
    "medium": "Model CPU offload: whole submodules swap between RAM and VRAM. Good balance.",
    "high": "Everything resident on the GPU. Fastest, needs the most VRAM.",
}


@dataclass(frozen=True)
class Hardware:
    """A snapshot of what we are running on."""

    device: str
    gpu_name: str
    vram_gb: float
    ram_gb: float
    profile: Profile
    torch_version: str
    cuda_version: str | None

    @property
    def has_cuda(self) -> bool:
        return self.device == "cuda"

    def summary(self) -> str:
        if not self.has_cuda:
            return f"CPU only ({self.ram_gb:.1f} GB RAM) - torch {self.torch_version}"
        return (
            f"{self.gpu_name} - {self.vram_gb:.1f} GB VRAM / {self.ram_gb:.1f} GB RAM "
            f"- torch {self.torch_version} (CUDA {self.cuda_version})"
        )


def _profile_for(vram_gb: float, has_cuda: bool) -> Profile:
    if not has_cuda:
        return "cpu"
    if vram_gb < LOW_VRAM_CEILING:
        return "low"
    if vram_gb < MEDIUM_VRAM_CEILING:
        return "medium"
    return "high"


def detect() -> Hardware:
    """Probe the machine once and derive a default memory profile."""
    ram_gb = psutil.virtual_memory().total / 1024**3
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        return Hardware(
            device="cpu",
            gpu_name="cpu",
            vram_gb=0.0,
            ram_gb=ram_gb,
            profile="cpu",
            torch_version=torch.__version__,
            cuda_version=None,
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    return Hardware(
        device="cuda",
        gpu_name=props.name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        profile=_profile_for(vram_gb, True),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def free_memory() -> None:
    """Drop cached allocations. Call after unloading a pipeline."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_in_use_gb() -> float:
    """Currently reserved VRAM, for the live readout in the sidebar."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / 1024**3


def resolve_dtype(name: str, hw: Hardware) -> torch.dtype:
    """Map a spec's dtype string to a torch dtype the device can actually use.

    Ampere (RTX 30xx) and newer support bf16; older cards silently fall back to
    fp16, and CPU runs stay in fp32 because fp16 matmuls there are glacial.
    """
    if not hw.has_cuda:
        return torch.float32
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("bf16 unsupported on %s - falling back to fp16", hw.gpu_name)
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.float16
