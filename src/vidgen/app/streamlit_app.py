"""Streamlit front end for the local video generation studio.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# Streamlit executes this file as a script rather than importing it as a module,
# so `src` has to be on the path before the package imports resolve.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from vidgen import hardware
from vidgen.app.components import (
    gallery,
    hardware_panel,
    model_card,
    profile_selector,
    reference_uploader,
)
from vidgen.backends import GenerationRequest
from vidgen.config import DEFAULT_NEGATIVE_PROMPT, MAX_REFERENCE_IMAGES, OUTPUT_DIR
from vidgen.manager import ModelManager
from vidgen.registry import ModelSpec, all_specs, get as get_spec
from vidgen.utils.cache import spec_is_cached
from vidgen.utils.logging import setup_logging
from vidgen.utils.video import list_outputs

setup_logging()

st.set_page_config(
    page_title="vidgen · local video studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_PROMPTS = [
    "A golden retriever puppy running through tall grass at sunset, "
    "shallow depth of field, warm rim light, slow motion",
    "Aerial drone shot flying over a misty pine forest at dawn, "
    "volumetric light rays, cinematic",
    "Close-up of coffee being poured into a glass cup, steam rising, "
    "soft window light, macro lens",
    "A neon-lit Tokyo alley in the rain, reflections on wet asphalt, "
    "handheld camera, cyberpunk",
]


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_manager() -> ModelManager:
    """One manager for the whole server process - it holds the loaded weights."""
    return ModelManager()


@st.cache_resource(show_spinner=False)
def get_hardware():
    return hardware.detect()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar(hw) -> tuple[ModelSpec, dict]:
    st.sidebar.title("🎬 vidgen")
    st.sidebar.caption("Local open-source video generation")

    specs = all_specs()
    labels = {spec.key: spec.name for spec in specs}
    model_key = st.sidebar.selectbox(
        "Model",
        [spec.key for spec in specs],
        format_func=lambda key: labels[key],
        help="Listed lightest-first. AnimateDiff is the safest choice on a small GPU.",
    )
    spec = get_spec(model_key)
    model_card(spec, hw)

    st.sidebar.divider()
    hardware_panel(hw)
    profile = profile_selector(hw)

    st.sidebar.divider()
    st.sidebar.subheader("Output")

    if spec.lock_resolution:
        width, height = spec.default_resolution
        st.sidebar.info(f"Resolution fixed at {width}×{height} by this checkpoint.")
    else:
        # Labels rather than (w, h) tuples: select_slider reads a tuple `value`
        # as a two-handle range and would look up 512 and 320 separately.
        by_label = {f"{w}×{h}": (w, h) for w, h in spec.resolutions}
        choice = st.sidebar.select_slider(
            "Resolution",
            options=list(by_label),
            value=f"{spec.default_resolution[0]}×{spec.default_resolution[1]}",
        )
        width, height = by_label[choice]

    if spec.lock_frames:
        frames = spec.default_frames
        st.sidebar.info(f"Frame count fixed at {frames} by this checkpoint.")
    else:
        frames = st.sidebar.slider(
            "Frames",
            min_value=spec.frame_min,
            max_value=spec.frame_max,
            value=spec.default_frames,
            step=spec.frame_step,
        )
        frames = spec.snap_frames(frames)

    fps = st.sidebar.slider("Playback FPS", 4, 30, spec.default_fps)
    st.sidebar.caption(
        f"≈ {spec.duration_seconds(frames, fps):.1f} s of video "
        f"({frames} frames at {fps} fps)"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Sampling")

    steps = st.sidebar.slider(
        "Inference steps",
        spec.steps_range[0],
        spec.steps_range[1],
        spec.default_steps,
        help="More steps means more detail and a proportionally longer render.",
    )
    guidance = st.sidebar.slider(
        "Guidance scale",
        spec.guidance_range[0],
        spec.guidance_range[1],
        spec.default_guidance,
        step=0.5,
        help="How strictly the model follows the prompt. Too high looks over-saturated.",
    )

    seed_col, lock_col = st.sidebar.columns([3, 2])
    random_seed = lock_col.checkbox("Random", value=True)
    seed = seed_col.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=42,
        disabled=random_seed,
    )

    settings = {
        "profile": profile,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "guidance": guidance,
        "seed": -1 if random_seed else int(seed),
    }

    with st.sidebar.expander("Advanced"):
        settings["reference_strength"] = st.slider(
            "Reference strength",
            0.0,
            1.0,
            0.7,
            step=0.05,
            help="IP-Adapter weight. Only used by models with style-reference support.",
        )
        if spec.backend == "svd":
            settings["motion_bucket_id"] = st.slider(
                "Motion amount", 1, 255, 127,
                help="SVD-specific: higher means more movement.",
            )
            settings["noise_aug_strength"] = st.slider(
                "Noise augmentation", 0.0, 1.0, 0.02, step=0.01,
                help="Higher lets the clip drift further from the reference image.",
            )
        settings["keep_loaded"] = st.checkbox(
            "Keep model loaded after render",
            value=True,
            help="Uncheck to free VRAM immediately once a clip finishes.",
        )

    st.sidebar.divider()
    if st.sidebar.button("Unload model / free VRAM", use_container_width=True):
        get_manager().unload()
        st.sidebar.success("Unloaded.")

    return spec, settings


# --------------------------------------------------------------------------- #
# Generate tab
# --------------------------------------------------------------------------- #


def _apply_example(text: str) -> None:
    st.session_state["prompt"] = text


def render_generate_tab(spec: ModelSpec, settings: dict) -> None:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("#### Prompt")

        if spec.backend == "svd":
            st.info(
                "Stable Video Diffusion has no text encoder — it animates the "
                "reference image and ignores the prompt entirely."
            )

        # The example buttons write straight into session state, so the widget
        # must be driven by `key` alone - passing `value` too is a Streamlit error.
        st.session_state.setdefault("prompt", EXAMPLE_PROMPTS[0])
        prompt = st.text_area(
            "Describe the shot",
            height=140,
            key="prompt",
            label_visibility="collapsed",
            placeholder="A cinematic shot of ...",
            disabled=spec.backend == "svd",
        )

        # on_click runs before the rerun, which is the only point at which a
        # widget-backed session_state key may still be written.
        example_cols = st.columns(len(EXAMPLE_PROMPTS))
        for index, (column, example) in enumerate(zip(example_cols, EXAMPLE_PROMPTS)):
            column.button(
                f"Example {index + 1}",
                use_container_width=True,
                on_click=_apply_example,
                args=(example,),
                key=f"example_{index}",
            )

        if spec.supports_negative_prompt:
            negative = st.text_area(
                "Negative prompt",
                value=spec.default_negative,
                height=80,
                help="Concepts to steer away from.",
            )
        else:
            negative = ""
            st.caption("This model does not take a negative prompt.")

    with right:
        references = reference_uploader(spec, MAX_REFERENCE_IMAGES)

    st.divider()

    estimated = settings["steps"] * settings["frames"]
    st.caption(
        f"About to render **{settings['width']}×{settings['height']}**, "
        f"{settings['frames']} frames, {settings['steps']} steps "
        f"(~{estimated:,} step-frames of work) on the "
        f"`{settings['profile']}` memory profile."
    )

    if st.button("🎬 Generate video", type="primary", use_container_width=True):
        run_generation(spec, settings, prompt, negative, references)

    _render_last_result()


def run_generation(
    spec: ModelSpec,
    settings: dict,
    prompt: str,
    negative: str,
    references: list,
) -> None:
    manager = get_manager()
    manager.with_profile(settings["profile"])

    extra = {
        key: settings[key]
        for key in ("motion_bucket_id", "noise_aug_strength")
        if key in settings
    }

    request = GenerationRequest(
        prompt=prompt.strip(),
        negative_prompt=negative.strip(),
        reference_images=references,
        width=settings["width"],
        height=settings["height"],
        num_frames=settings["frames"],
        fps=settings["fps"],
        steps=settings["steps"],
        guidance_scale=settings["guidance"],
        seed=settings["seed"],
        reference_strength=settings.get("reference_strength", 0.7),
        extra=extra,
    )

    progress_bar = st.progress(0.0, text="Preparing…")
    status = st.empty()
    started = time.perf_counter()

    def on_progress(current: int, total: int, message: str) -> None:
        fraction = 0.0 if total <= 0 else min(current / total, 1.0)
        elapsed = time.perf_counter() - started
        suffix = ""
        if current > 0 and fraction > 0:
            eta = elapsed / fraction - elapsed
            suffix = f" · {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
        progress_bar.progress(fraction, text=f"{message}{suffix}")

    # Say what is actually about to happen. A blanket "this also downloads the
    # weights" on every render is wrong once the model is cached, and it hides
    # the case where a 14 GB download really is about to start.
    if not spec_is_cached(spec):
        spinner_text = (
            f"Downloading {spec.name} (~{spec.approx_download_gb:.0f} GB), then rendering. "
            "This only happens once."
        )
    elif manager.active_key != spec.key:
        spinner_text = f"Loading {spec.name} from disk, then rendering…"
    else:
        spinner_text = "Rendering…"

    try:
        with st.spinner(spinner_text):
            result, path = manager.render(spec.key, request, progress=on_progress)
    except ValueError as exc:
        progress_bar.empty()
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - the UI is the only place to report this
        progress_bar.empty()
        st.error(f"Generation failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return

    progress_bar.progress(1.0, text="Done")
    status.success(
        f"Rendered {len(result.frames)} frames in {result.elapsed_seconds:.0f}s "
        f"({result.elapsed_seconds / max(len(result.frames), 1):.1f}s per frame) · seed {result.seed}"
    )

    if not settings.get("keep_loaded", True):
        manager.unload()

    st.session_state["last_video"] = str(path) if path else None
    st.session_state["last_metadata"] = result.metadata


def _render_last_result() -> None:
    video_path = st.session_state.get("last_video")
    if not video_path or not Path(video_path).exists():
        return

    st.divider()
    st.markdown("#### Result")
    video_col, meta_col = st.columns([3, 2], gap="large")

    with video_col:
        st.video(video_path)
        st.download_button(
            "Download MP4",
            data=Path(video_path).read_bytes(),
            file_name=Path(video_path).name,
            mime="video/mp4",
            use_container_width=True,
        )

    with meta_col:
        st.caption(f"Saved to `{video_path}`")
        st.json(st.session_state.get("last_metadata", {}), expanded=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    hw = get_hardware()
    spec, settings = render_sidebar(hw)

    st.title("Local video generation")
    st.caption(hw.summary())

    generate_tab, gallery_tab, about_tab = st.tabs(["Generate", "Gallery", "Models"])

    with generate_tab:
        render_generate_tab(spec, settings)

    with gallery_tab:
        header, refresh = st.columns([5, 1])
        header.markdown(f"#### Rendered clips &nbsp;·&nbsp; `{OUTPUT_DIR}`")
        if refresh.button("Refresh", use_container_width=True):
            st.rerun()
        gallery(list_outputs(limit=30))

    with about_tab:
        st.markdown("#### Catalogue")
        for model_spec in all_specs():
            with st.expander(f"{model_spec.name} · {model_spec.params}"):
                st.markdown(
                    f"- **Repository:** `{model_spec.repo}`\n"
                    f"- **License:** {model_spec.license}\n"
                    f"- **Modes:** {', '.join(model_spec.modes)}\n"
                    f"- **Approx. download:** {model_spec.approx_download_gb:.0f} GB\n"
                    f"- **Suggested VRAM:** {model_spec.recommended_vram_gb:.0f} GB\n"
                    f"- **Default:** {model_spec.default_resolution[0]}×"
                    f"{model_spec.default_resolution[1]}, {model_spec.default_frames} frames "
                    f"@ {model_spec.default_fps} fps"
                )
                if model_spec.notes:
                    st.info(model_spec.notes)


main()
