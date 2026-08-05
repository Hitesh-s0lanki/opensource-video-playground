"""Headless renderer - the same engine the Streamlit app drives.

    vidgen --list
    vidgen -m ltx-video -p "a paper boat drifting down a rain gutter"
    vidgen -m svd-xt --ref photo.jpg --frames 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config  # noqa: F401  - imported first for its env-var side effects
from .backends import GenerationRequest
from .manager import get_manager
from .registry import all_specs, get as get_spec
from .utils.images import load_image
from .utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidgen",
        description="Generate video locally with open-source diffusion models.",
    )
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    parser.add_argument("-m", "--model", default="ltx-video", help="Model key (see --list).")
    parser.add_argument("-p", "--prompt", default="", help="Text prompt.")
    parser.add_argument("-n", "--negative", default=None, help="Negative prompt.")
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="PATH",
        help="Reference image. Repeat for multiple.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=-1, help="-1 for random.")
    parser.add_argument(
        "--ref-strength", type=float, default=0.7, help="IP-Adapter weight (0-1)."
    )
    parser.add_argument(
        "--motion-bucket",
        type=int,
        default=None,
        metavar="1-255",
        help="SVD only: how much movement. Low values (30-80) give a gentle camera "
        "drift that keeps architecture rigid; high values warp geometry.",
    )
    parser.add_argument(
        "--noise-aug",
        type=float,
        default=None,
        metavar="0-1",
        help="SVD only: how far the clip may drift from the reference image. "
        "Keep low (0.0-0.05) to preserve the input faithfully.",
    )
    parser.add_argument(
        "--profile",
        choices=["low", "medium", "high", "cpu"],
        default=None,
        help="Override the auto-detected memory profile.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _print_catalogue() -> None:
    for spec in all_specs():
        print(f"\n{spec.key}")
        print(f"  {spec.name} · {spec.params}")
        print(f"  modes:    {', '.join(spec.modes)}")
        print(f"  repo:     {spec.repo}")
        print(f"  license:  {spec.license}")
        print(
            f"  default:  {spec.default_resolution[0]}x{spec.default_resolution[1]}, "
            f"{spec.default_frames} frames @ {spec.default_fps} fps"
        )
        print(f"  download: ~{spec.approx_download_gb:.0f} GB")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")

    if args.list:
        _print_catalogue()
        return 0

    try:
        spec = get_spec(args.model)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    references = []
    for path in args.ref:
        source = Path(path)
        if not source.exists():
            print(f"Reference image not found: {source}", file=sys.stderr)
            return 2
        references.append(load_image(source))

    extra: dict[str, float | int] = {}
    if args.motion_bucket is not None:
        extra["motion_bucket_id"] = args.motion_bucket
    if args.noise_aug is not None:
        extra["noise_aug_strength"] = args.noise_aug

    width, height = spec.default_resolution
    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative if args.negative is not None else spec.default_negative,
        reference_images=references,
        width=args.width or width,
        height=args.height or height,
        num_frames=args.frames or spec.default_frames,
        fps=args.fps or spec.default_fps,
        steps=args.steps or spec.default_steps,
        guidance_scale=args.guidance if args.guidance is not None else spec.default_guidance,
        seed=args.seed,
        reference_strength=args.ref_strength,
        extra=extra,
    )

    manager = get_manager()
    if args.profile:
        manager.with_profile(args.profile)

    last_step = -1

    def on_progress(current: int, total: int, message: str) -> None:
        nonlocal last_step
        if current != last_step:
            last_step = current
            print(f"\r  {message:<48}", end="", flush=True)

    try:
        result, path = manager.render(
            spec.key, request, progress=on_progress, output_dir=args.output_dir
        )
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\nGeneration failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n\nDone in {result.elapsed_seconds:.0f}s · seed {result.seed} · "
        f"{len(result.frames)} frames"
    )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
