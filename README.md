# vidgen — local open-source video generation

A Streamlit studio for running open-weight text-to-video and image-to-video
models entirely on your own machine. Five models behind one interface, with the
offload strategy chosen automatically from your GPU's VRAM.

Built and tuned for a **4 GB RTX 3050 Laptop / 16 GB RAM** class of machine — it
scales up cleanly to bigger cards, and the constraints that make it work on 4 GB
(sequential offload, VAE tiling, one resident model) are all switchable.

---

## Models

| Key | Model | Params | Modes | License | Download |
|---|---|---|---|---|---|
| `ltx-video` | LTX-Video (Lightricks) | 2B DiT | text→video, image→video | LTXV Open Weights | ~14 GB |
| `animatediff-sd15` | AnimateDiff + SD 1.5 | 0.9B + 0.4B | text→video, style reference | Apache-2.0 / OpenRAIL-M | ~7 GB |
| `cogvideox-2b` | CogVideoX-2B (Zhipu) | 2B DiT | text→video | Apache-2.0 | ~13 GB |
| `wan21-t2v-1.3b` | Wan 2.1 T2V-1.3B (Alibaba) | 1.3B DiT | text→video | Apache-2.0 | ~15 GB |
| `svd-xt` | Stable Video Diffusion XT | 1.5B U-Net | image→video | Stability Community | ~9.5 GB |

**Start with `animatediff-sd15`** on a 4 GB card. It is the smallest download,
the fastest render, and the only model here that fits without heroics. Move to
`ltx-video` once that works — it is a large quality jump.

### How reference images are used

The three modes are genuinely different and the UI tells you which applies:

- **image→video** (`ltx-video`, `svd-xt`) — your image becomes the literal first
  frame; the model animates forward from it.
- **style reference** (`animatediff-sd15`) — your image is CLIP-encoded and
  injected via IP-Adapter, so the clip inherits its subject and style while the
  *prompt* drives the motion. Multiple images blend together.
- **text→video only** (`cogvideox-2b`, `wan21-t2v-1.3b`) — no image conditioning
  path exists at these model sizes. The uploader is disabled with an explanation
  rather than silently ignoring your upload.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a recent driver.

```powershell
# one-shot: creates .venv, installs everything, verifies CUDA
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Or manually:

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e . --no-deps
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `False`, you have the CPU-only wheel. The
`--extra-index-url` at the top of `requirements.txt` is what prevents that; swap
`cu124` for `cu126` or `cu121` if your driver needs a different CUDA build.

> **Use `uv run --no-sync`, not bare `uv run`.** Bare `uv run` re-resolves the
> environment against `pyproject.toml` on every invocation. If the checkout sits
> in a cloud-synced folder (OneDrive, Dropbox), a file locked by the sync client
> mid-reinstall leaves that package half-uninstalled — this broke numpy during
> development. `--no-sync` skips the re-resolve, and calling `.venv\Scripts\...`
> directly avoids it entirely.

### Verify the install

```powershell
.\.venv\Scripts\python.exe scripts\selftest.py
```

Downloads nothing. It checks the CUDA build, constructs all five backends,
asserts each model's frame/resolution constraints actually hold, exercises
reference-image preparation and MP4 export, and runs the Streamlit script end to
end for every model. Run it first when something misbehaves — it separates "my
environment is broken" from "the model ran out of memory".

### Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Opens on <http://localhost:8501>.

### Pre-download weights (optional)

The first render of a model also downloads it. To get that out of the way:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --model animatediff-sd15
.\.venv\Scripts\python.exe scripts\download_models.py --all   # ~58 GB, all five
```

Weights land in `models/` (this project sets `HF_HOME` there) rather than
silently filling `C:\Users\<you>\.cache`.

### Headless CLI

```powershell
.\.venv\Scripts\vidgen.exe --list
.\.venv\Scripts\vidgen.exe -m ltx-video -p "a paper boat drifting down a rain gutter"
.\.venv\Scripts\vidgen.exe -m svd-xt --ref photo.jpg
.\.venv\Scripts\vidgen.exe -m animatediff-sd15 -p "a fox in snow" --ref style.png --ref-strength 0.8
```

#### Animating a still photo (interiors, products, architecture)

SVD is the right model when the subject must stay exactly as photographed and you
only want the camera to move. Its two controls matter more than anything else:

```powershell
.\.venv\Scripts\vidgen.exe -m svd-xt --ref room.jpg `
  --width 1024 --height 576 --motion-bucket 60 --noise-aug 0.02
```

- `--motion-bucket` (1-255) — how much movement. **30-80 for architecture**: a
  gentle camera drift with rigid walls and furniture. Above ~150 the model starts
  inventing motion and geometry visibly bends.
- `--noise-aug` (0-1) — how far the clip may depart from the reference. Keep at
  **0.0-0.05** to stay faithful; higher values let it reinterpret the scene.

Match the aspect ratio of the source image to avoid cropping, and prefer 1024×576
where memory allows — SVD is trained there and softens noticeably below it.

---

## Memory profiles

The single most important setting, auto-detected from your VRAM and overridable
in the sidebar:

| Profile | Strategy | Picked when |
|---|---|---|
| `low` | Sequential CPU offload — weights stream to the GPU block by block | < 6.5 GB VRAM |
| `medium` | Model CPU offload — one whole submodule at a time | 6.5–13 GB |
| `high` | Everything resident on the GPU | ≥ 13 GB |

On 4 GB you are on `low`. It works, and it is slow: expect a couple of minutes
per clip from AnimateDiff and considerably longer from the DiT models. That cost
is PCIe transfer, not computation.

### If you hit out-of-memory

In order of what to try:

1. Drop the resolution one notch, then the frame count.
2. Switch to `animatediff-sd15`.
3. Confirm the profile is `low`.
4. Raise your Windows page file to ~32 GB — under sequential offload the model
   lives in system RAM, and 16 GB is tight for the T5-XXL text encoders that
   LTX, CogVideoX and Wan all ship.

---

## Layout

```
video-generation/
├── app.py                      # streamlit run app.py
├── pyproject.toml              # uv-managed deps + the CUDA torch index
├── requirements.txt            # plain-pip mirror of the same deps
├── scripts/
│   ├── setup.ps1               # venv + install + CUDA check
│   ├── selftest.py             # verify the install, downloads nothing
│   └── download_models.py      # weight prefetcher
├── outputs/                    # rendered .mp4 + .json metadata sidecars
└── src/vidgen/
    ├── config.py               # paths, env tuning, default prompts
    ├── hardware.py             # GPU detection -> memory profile
    ├── registry.py             # ModelSpec catalogue (add models here)
    ├── manager.py              # single-resident-model lifecycle + render()
    ├── cli.py                  # headless entry point
    ├── backends/
    │   ├── base.py             # lifecycle, offload strategy, safe pipe calls
    │   ├── ltx.py  cogvideox.py  wan.py  svd.py  animatediff.py
    ├── utils/                  # image prep, video export, logging
    └── app/                    # Streamlit UI
```

### Adding a model

Two steps: append a `ModelSpec` to `src/vidgen/registry.py` describing its valid
resolutions, frame arithmetic and sampler ranges, then add a `VideoBackend`
subclass implementing `_build()` and `_generate_frames()`. The UI builds its
controls from the spec, so nothing in the interface needs touching.

---

## Notes and caveats

- **Frame counts are constrained by architecture.** LTX needs `8n+1` frames and
  dimensions that are multiples of 32; CogVideoX is locked to 720×480; SVD is
  locked to 25 frames. The UI snaps your input onto each model's valid grid
  rather than letting the pipeline fail deep in a forward pass.
- **Licenses differ and two are not Apache-2.0.** CogVideoX-2B and Wan 2.1 are
  Apache-2.0. LTX-Video permits commercial use below a revenue cap. SVD is
  non-commercial above $1M revenue. Check the terms before shipping output
  commercially — the exact license is shown in the sidebar and recorded in every
  metadata sidecar.
- **Every render writes a `.json` sidecar** next to the `.mp4` with the model,
  seed and every sampler setting, so a clip you like is reproducible.
