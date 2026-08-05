"""Project-wide paths, tunables and environment setup.

Importing this module has two side effects: it loads ``.env`` from the project
root, and it sets a few CUDA/HF environment variables that only take effect if
they are set *before* torch is imported. Every entry point (`app.py`, `cli.py`,
the scripts) imports this first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #

#: True when a .env file was found and loaded (surfaced by scripts/selftest.py).
ENV_FILE_LOADED = False

try:
    from dotenv import load_dotenv

    # override=False: a variable already set in the real environment beats the
    # file, so `set VIDGEN_MODEL_DIR=...` for one run still works as expected.
    # An explicit path rather than find_dotenv() so the file is located relative
    # to the project, not to whatever directory the app was launched from.
    ENV_FILE_LOADED = load_dotenv(dotenv_path=ENV_FILE, override=False)
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    # Not fatal: every setting below has a working default, and real environment
    # variables are still honoured.
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def _path_from_env(name: str, default: Path) -> Path:
    """Read a path setting, ignoring blank values left behind in a .env file."""
    raw = os.environ.get(name, "").strip().strip('"').strip("'")
    return Path(raw).expanduser() if raw else default


OUTPUT_DIR = _path_from_env("VIDGEN_OUTPUT_DIR", PROJECT_ROOT / "outputs")
MODEL_DIR = _path_from_env("VIDGEN_MODEL_DIR", PROJECT_ROOT / "models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Keep the (very large) Hugging Face cache inside the project unless the user
# has already pointed HF_HOME somewhere else. Video checkpoints are 5-40 GB;
# having them land silently in C:\Users\<you>\.cache is a bad surprise.
os.environ.setdefault("HF_HOME", str(MODEL_DIR))

# --------------------------------------------------------------------------- #
# Memory / allocator tuning
# --------------------------------------------------------------------------- #

# Expandable segments dramatically reduce fragmentation-driven OOMs when the
# whole point of the run is streaming blocks on and off a small card. The
# allocator does not implement them on Windows, where setting it only produces a
# warning on every tensor move, so it is applied per-platform.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Tokenizers' parallelism warning is noise inside Streamlit's rerun model.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Windows cannot create symlinks without Developer Mode or admin rights.
# huggingface_hub probes for support and falls back to copying, but the probe is
# unreliable under a filesystem filter driver (OneDrive) and with concurrent
# workers - when it guesses wrong the download dies with
# "WinError 1314: A required privilege is not held by the client".
# Disabling symlinks outright makes the copy path unconditional and removes the
# failure mode. It costs disk (blobs are duplicated into the snapshot), which is
# the same cost the fallback already pays.
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --------------------------------------------------------------------------- #
# Generation defaults
# --------------------------------------------------------------------------- #

#: Reused by most backends that take a negative prompt.
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, static image, deformed limbs"
)

#: Wan's reference implementation ships this Chinese negative prompt and the
#: model responds noticeably better to it than to an English translation.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "杂乱的背景,三条腿,背景人很多,倒着走"
)

#: Hard ceiling on uploaded reference images kept per request.
MAX_REFERENCE_IMAGES = int(os.environ.get("VIDGEN_MAX_REFERENCE_IMAGES", "4"))

#: Video container written by the exporter.
VIDEO_SUFFIX = ".mp4"


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #


def describe() -> dict[str, str]:
    """The effective configuration, for the self-test and the app sidebar.

    Never reports the value of ``HF_TOKEN`` - only whether one is present.
    """
    on_onedrive = "OneDrive" in str(MODEL_DIR)
    return {
        ".env file": str(ENV_FILE) if ENV_FILE_LOADED else "not found (using defaults)",
        "model cache": str(MODEL_DIR) + ("   <-- inside OneDrive" if on_onedrive else ""),
        "output dir": str(OUTPUT_DIR),
        "HF_HOME": os.environ.get("HF_HOME", "unset"),
        "HF_TOKEN": "set" if os.environ.get("HF_TOKEN") else "not set (downloads are rate-limited)",
        "log level": os.environ.get("VIDGEN_LOG_LEVEL", "INFO"),
        "max reference images": str(MAX_REFERENCE_IMAGES),
    }
