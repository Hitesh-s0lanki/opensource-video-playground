"""Verify the install without downloading a single model weight.

    .\.venv\Scripts\python.exe scripts\selftest.py

Checks the CUDA build, constructs every backend, asserts each model's frame and
resolution constraints actually hold, exercises reference-image preparation and
MP4 export, and runs the Streamlit script end to end. Run this first when
something misbehaves - it isolates "my environment is broken" from "the model
ran out of memory".
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from PIL import Image  # noqa: E402

from vidgen import hardware  # noqa: E402
from vidgen.backends import GenerationRequest, create  # noqa: E402
from vidgen.registry import all_specs, get as get_spec  # noqa: E402
from vidgen.utils.images import prepare_reference  # noqa: E402
from vidgen.utils.logging import setup_logging  # noqa: E402
from vidgen.utils.video import list_outputs, save_video, slugify  # noqa: E402

setup_logging("WARNING")

PASS, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"
failures: list[str] = []
warnings: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """A hard requirement. Failing one means the install is broken."""
    if condition:
        print(f"{PASS} {label}" + (f" - {detail}" if detail else ""))
    else:
        print(f"{FAIL} {label}" + (f" - {detail}" if detail else ""))
        failures.append(label)


def warn(label: str, condition: bool, detail: str = "") -> None:
    """Advisory only - the app still works, but something is worth fixing."""
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{WARN} {label}" + (f" - {detail}" if detail else ""))
        warnings.append(label)


# --------------------------------------------------------------------------- #
print("\n== Environment ==")
# --------------------------------------------------------------------------- #

import torch  # noqa: E402

from vidgen import config  # noqa: E402

for key, value in config.describe().items():
    print(f"  {key:<22} {value}")
print()

warn(
    "model cache is outside a cloud-synced folder",
    not any(p in str(config.MODEL_DIR) for p in ("OneDrive", "Dropbox", "Google Drive")),
    "weights here get uploaded to the cloud and can be locked mid-write; "
    "move them and set VIDGEN_MODEL_DIR in .env",
)

hw = hardware.detect()
print(f"  {hw.summary()}")
check("torch imports", True, torch.__version__)
check(
    "CUDA available",
    hw.has_cuda,
    "generation will run on CPU and be extremely slow" if not hw.has_cuda else hw.gpu_name,
)
check("memory profile resolved", hw.profile in ("cpu", "low", "medium", "high"), hw.profile)

try:
    from diffusers import (  # noqa: F401
        AnimateDiffPipeline,
        AutoencoderKLWan,
        CogVideoXPipeline,
        LTXImageToVideoPipeline,
        LTXPipeline,
        MotionAdapter,
        StableVideoDiffusionPipeline,
        WanPipeline,
    )

    check("all diffusers pipeline classes present", True)
except ImportError as exc:
    check("all diffusers pipeline classes present", False, str(exc))

# --------------------------------------------------------------------------- #
print("\n== Backends and constraint handling ==")
# --------------------------------------------------------------------------- #

# A deliberately invalid request: off-grid dimensions, absurd frames/steps/cfg.
oversized = Image.new("RGB", (1920, 1080), (10, 120, 200))

for spec in all_specs():
    try:
        backend = create(spec, hw)
    except Exception as exc:  # noqa: BLE001
        check(f"{spec.key}: construct", False, str(exc))
        continue

    request = GenerationRequest(
        prompt="a test shot",
        negative_prompt=spec.default_negative,
        reference_images=[oversized] if spec.uses_reference_images else [],
        width=1234,
        height=777,
        num_frames=999,
        steps=999,
        guidance_scale=99.0,
    )
    norm = backend.normalize(request)

    problems = []
    if norm.num_frames != spec.snap_frames(norm.num_frames):
        problems.append("frame count off-grid")
    if not spec.frame_min <= norm.num_frames <= spec.frame_max:
        problems.append("frame count out of range")
    if spec.lock_resolution:
        if (norm.width, norm.height) != spec.default_resolution:
            problems.append("locked resolution not enforced")
    elif norm.width % spec.dim_multiple or norm.height % spec.dim_multiple:
        problems.append(f"dimensions not multiples of {spec.dim_multiple}")
    if not spec.steps_range[0] <= norm.steps <= spec.steps_range[1]:
        problems.append("steps out of range")
    if not spec.guidance_range[0] <= norm.guidance_scale <= spec.guidance_range[1]:
        problems.append("guidance out of range")
    if spec.uses_reference_images:
        if norm.reference_images[0].size != (norm.width, norm.height):
            problems.append("reference not resized to target")
    elif norm.reference_images:
        problems.append("reference not dropped for a text-only model")

    check(
        f"{spec.key}: constraints",
        not problems,
        "; ".join(problems)
        or f"{norm.width}x{norm.height}, {norm.num_frames}f, {norm.steps} steps",
    )

# --------------------------------------------------------------------------- #
print("\n== Spec self-consistency ==")
# --------------------------------------------------------------------------- #

# Every resolution a spec *offers* must survive its own snapping untouched.
# Without this, the UI can present a choice the model would reject - which is
# how SVD ended up offering 512x288 when it needs multiples of 64.
for spec in all_specs():
    bad = [
        f"{w}x{h}"
        for w, h in spec.resolutions
        if (spec.snap_dimension(w), spec.snap_dimension(h)) != (w, h)
    ]
    check(
        f"{spec.key}: offered resolutions are multiples of {spec.dim_multiple}",
        not bad,
        ", ".join(bad),
    )
    check(
        f"{spec.key}: default resolution is offered",
        spec.default_resolution in spec.resolutions or spec.lock_resolution,
        str(spec.default_resolution),
    )
    check(
        f"{spec.key}: default frame count is on-grid",
        spec.snap_frames(spec.default_frames) == spec.default_frames,
        f"{spec.default_frames} -> {spec.snap_frames(spec.default_frames)}",
    )
    check(
        f"{spec.key}: defaults are inside their ranges",
        spec.steps_range[0] <= spec.default_steps <= spec.steps_range[1]
        and spec.guidance_range[0] <= spec.default_guidance <= spec.guidance_range[1],
    )
    check(
        f"{spec.key}: has a backend implementation",
        bool(spec.modes) and spec.backend in {s.backend for s in all_specs()},
    )

# --------------------------------------------------------------------------- #
print("\n== Architectural frame arithmetic ==")
# --------------------------------------------------------------------------- #

ltx = get_spec("ltx-video")
check(
    "LTX enforces 8n+1 frames",
    all((ltx.snap_frames(n) - 1) % 8 == 0 for n in range(1, 400)),
)
check("SVD frame count is locked to 25", get_spec("svd-xt").snap_frames(100) == 25)
check(
    "CogVideoX resolution is locked to 720x480",
    get_spec("cogvideox-2b").lock_resolution,
)

# --------------------------------------------------------------------------- #
print("\n== Validation ==")
# --------------------------------------------------------------------------- #

try:
    create(get_spec("svd-xt"), hw).validate(GenerationRequest(prompt="hello"))
    check("SVD rejects a missing reference image", False, "no error raised")
except ValueError:
    check("SVD rejects a missing reference image", True)

try:
    create(get_spec("wan21-t2v-1.3b"), hw).validate(GenerationRequest(prompt="  "))
    check("empty prompt rejected", False, "no error raised")
except ValueError:
    check("empty prompt rejected", True)

# --------------------------------------------------------------------------- #
print("\n== Reference image preparation ==")
# --------------------------------------------------------------------------- #

for fit in ("cover", "contain", "stretch"):
    out = prepare_reference(oversized, 512, 320, fit=fit)
    check(f"resize 1920x1080 -> 512x320 ({fit})", out.size == (512, 320), str(out.size))

# --------------------------------------------------------------------------- #
print("\n== Video export ==")
# --------------------------------------------------------------------------- #

frames = [Image.new("RGB", (128, 96), (i * 12 % 256, 60, 180)) for i in range(12)]
try:
    path = save_video(
        frames, fps=8, metadata={"model": "selftest", "prompt": "selftest"},
        stem_hint="selftest",
    )
    check("MP4 encoded", path.exists() and path.stat().st_size > 0, f"{path.stat().st_size:,} bytes")
    check("metadata sidecar written", path.with_suffix(".json").exists())
    check("gallery lists the new clip", any(r.path == path for r in list_outputs()))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
except Exception as exc:  # noqa: BLE001
    check("MP4 encoded", False, f"{exc} (is imageio-ffmpeg installed?)")

check("prompt slugified for filenames", slugify("A dog!! running / fast") == "a-dog-running-fast")

# --------------------------------------------------------------------------- #
print("\n== Streamlit app ==")
# --------------------------------------------------------------------------- #

try:
    from streamlit.testing.v1 import AppTest

    app_file = _ROOT / "src" / "vidgen" / "app" / "streamlit_app.py"
    for spec in all_specs():
        at = AppTest.from_file(str(app_file), default_timeout=120)
        at.run()
        at.sidebar.selectbox[0].set_value(spec.name)
        at.run()
        errors = [e.value for e in at.exception]
        check(f"UI renders for {spec.key}", not errors, errors[0] if errors else "")
except Exception as exc:  # noqa: BLE001
    check("Streamlit app renders", False, str(exc))

# --------------------------------------------------------------------------- #
print("\n== Streamlit HTTP stack ==")
# --------------------------------------------------------------------------- #

# AppTest above runs the script in-process and never starts a web server, so it
# cannot see middleware faults. Serving real HTTP requests - crucially
# advertising gzip, the way every browser does - is the only way to catch them.
#
# Probe `/_stcore/health` and not just `/`. Under a starlette/streamlit version
# mismatch the page itself still returns 200 while the health endpoint the
# frontend polls returns 500, which surfaces as the browser's "Connection failed
# with status 500" banner over an otherwise rendered page. Checking only `/`
# reports a healthy app that is, from the browser's point of view, broken.
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


port = _free_port()
server = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", str(_ROOT / "app.py"),
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    cwd=str(_ROOT),
    text=True,
)

BROWSER_HEADERS = {
    "Accept-Encoding": "gzip, deflate, br",
    "User-Agent": "Mozilla/5.0",
}


def _fetch(path: str, retries: int = 1) -> int | str:
    """GET a path as a browser would. Returns the status code, or a message."""
    for _ in range(retries):
        if server.poll() is not None:
            return f"server exited with code {server.returncode}"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", headers=BROWSER_HEADERS
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code  # a real reply, just an error one - stop retrying
        except Exception:
            time.sleep(1)
    return "no response"


try:
    page = _fetch("/", retries=60)  # doubles as waiting for the server to boot
    check("serves the page to a gzip-accepting browser", page == 200, f"HTTP {page}")

    health = _fetch("/_stcore/health")
    check(
        "serves /_stcore/health to a gzip-accepting browser",
        health == 200,
        f"HTTP {health}"
        + (
            "  <-- gzip middleware fault. The page loads but the browser shows "
            "'Connection failed with status 500'. Check the starlette/streamlit "
            "version pair: streamlit allows starlette<2, but starlette 1.x "
            "breaks its gzip middleware. Pin starlette<1.0."
            if health == 500
            else ""
        ),
    )
finally:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        server.kill()

# --------------------------------------------------------------------------- #
print()
if failures:
    print(f"FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    raise SystemExit(1)
if warnings:
    print(f"{len(warnings)} advisory warning(s): {', '.join(warnings)}\n")
print("All self-tests passed. Run a real generation with:")
print('  .\\.venv\\Scripts\\vidgen.exe -m animatediff-sd15 -p "a fox running in snow"')
raise SystemExit(0)
