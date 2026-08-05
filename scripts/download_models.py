"""Pre-download model weights so the first render is not also a 15 GB download.

    uv run python scripts/download_models.py --model animatediff-sd15
    uv run python scripts/download_models.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidgen import config  # noqa: E402,F401 - sets HF_HOME before hub import
from vidgen.registry import all_specs, get as get_spec  # noqa: E402
from vidgen.utils.logging import setup_logging  # noqa: E402


def fetch(repo: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"  -> {repo}")
    snapshot_download(
        repo_id=repo,
        # .bin duplicates the safetensors weights in most of these repos.
        ignore_patterns=["*.msgpack", "*.h5", "*.ckpt", "*.pth"],
        max_workers=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch model weights.")
    parser.add_argument("--model", action="append", default=[], help="Model key. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Download every catalogued model.")
    args = parser.parse_args()

    setup_logging()

    if args.all:
        specs = all_specs()
    elif args.model:
        specs = [get_spec(key) for key in args.model]
    else:
        parser.error("Pass --model <key> or --all. Use `vidgen --list` to see keys.")
        return 2

    total = sum(spec.approx_download_gb for spec in specs)
    print(f"\nDownloading {len(specs)} model(s), roughly {total:.0f} GB total.")
    print(f"Cache: {config.MODEL_DIR}\n")

    for spec in specs:
        print(f"{spec.name}")
        for repo in (spec.repo, *spec.extra_repos):
            try:
                fetch(repo)
            except Exception as exc:  # noqa: BLE001
                print(f"     failed: {exc}", file=sys.stderr)
                return 1
        print()

    print("All downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
