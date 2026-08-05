"""Reusable Streamlit widgets for the studio UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..hardware import Hardware, PROFILE_DESCRIPTIONS, vram_in_use_gb
from ..registry import ModelSpec
from ..utils.cache import cached_size_gb, spec_is_cached
from ..utils.images import load_image, to_thumbnail
from ..utils.video import OutputRecord

MODE_BADGES = {
    "t2v": "text-to-video",
    "i2v": "image-to-video",
    "ref": "style reference",
}


def hardware_panel(hw: Hardware) -> None:
    """Sidebar readout of what we detected and what that implies."""
    st.sidebar.subheader("Hardware")
    if hw.has_cuda:
        st.sidebar.caption(hw.gpu_name)
        col_a, col_b = st.sidebar.columns(2)
        col_a.metric("VRAM", f"{hw.vram_gb:.1f} GB")
        col_b.metric("RAM", f"{hw.ram_gb:.1f} GB")
        in_use = vram_in_use_gb()
        if in_use > 0.05:
            st.sidebar.progress(
                min(in_use / hw.vram_gb, 1.0),
                text=f"{in_use:.1f} / {hw.vram_gb:.1f} GB reserved",
            )
    else:
        st.sidebar.error("No CUDA device detected - generation will run on CPU.")
        st.sidebar.metric("RAM", f"{hw.ram_gb:.1f} GB")


def profile_selector(hw: Hardware) -> str:
    """Let the user override the auto-detected offload strategy."""
    options = ["low", "medium", "high"] if hw.has_cuda else ["cpu"]
    default_index = options.index(hw.profile) if hw.profile in options else 0

    profile = st.sidebar.selectbox(
        "Memory profile",
        options,
        index=default_index,
        help="How aggressively weights are offloaded to system RAM.",
    )
    st.sidebar.caption(PROFILE_DESCRIPTIONS[profile])
    if hw.has_cuda and profile != hw.profile:
        st.sidebar.warning(
            f"Auto-detection chose `{hw.profile}` for {hw.vram_gb:.1f} GB of VRAM. "
            "A lighter profile may run out of memory."
        )
    return profile


def model_card(spec: ModelSpec, hw: Hardware) -> None:
    """The descriptive block shown under the model picker."""
    badges = " · ".join(MODE_BADGES[m] for m in spec.modes)
    st.sidebar.markdown(f"**{spec.params}** — {badges}")

    # Whether the weights are already local is the single biggest factor in how
    # long the next click takes, so it belongs on the card, not buried in a log.
    on_disk = spec_is_cached(spec)
    col_a, col_b = st.sidebar.columns(2)
    if on_disk:
        col_a.metric("Weights", f"{cached_size_gb(spec):.1f} GB", "on disk")
    else:
        col_a.metric("Weights", f"~{spec.approx_download_gb:.0f} GB", "to download")
    col_b.metric("Suggested VRAM", f"{spec.recommended_vram_gb:.0f} GB")

    if not on_disk:
        st.sidebar.info(
            f"First render downloads ~{spec.approx_download_gb:.0f} GB before it starts. "
            "One time only."
        )

    if hw.has_cuda and hw.vram_gb < spec.recommended_vram_gb:
        st.sidebar.warning(
            f"Below the suggested {spec.recommended_vram_gb:.0f} GB. It should still "
            "run on the `low` profile, but expect long renders."
        )

    with st.sidebar.expander("Model details"):
        st.markdown(f"**Repository** &nbsp;`{spec.repo}`")
        st.markdown(f"**License** &nbsp;{spec.license}")
        if spec.notes:
            st.caption(spec.notes)


def reference_uploader(spec: ModelSpec, max_images: int) -> list[Any]:
    """Upload widget plus previews. Returns a list of PIL images."""
    st.markdown("#### Reference images")

    if not spec.uses_reference_images:
        st.info(
            f"{spec.name} is text-to-video only — it has no image conditioning path, "
            "so any uploads here would be ignored. Pick LTX-Video, SVD, or AnimateDiff "
            "to use a reference image."
        )
        return []

    if spec.supports_i2v:
        st.caption("The first image becomes the literal first frame of the clip.")
    else:
        st.caption(
            "Images guide style and subject through IP-Adapter — they do not become "
            "frame 0. Multiple images are blended together."
        )

    uploads = st.file_uploader(
        "PNG / JPG / WEBP",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if not uploads:
        if spec.modes == ("i2v",):
            st.warning("This model requires a reference image.")
        return []

    if len(uploads) > max_images:
        st.warning(f"Using the first {max_images} images; the rest were dropped.")
        uploads = uploads[:max_images]

    images = []
    for upload in uploads:
        try:
            images.append(load_image(upload))
        except Exception as exc:  # noqa: BLE001 - surface any decode failure to the user
            st.error(f"Could not read {upload.name}: {exc}")

    if images:
        for column, image in zip(st.columns(len(images)), images):
            column.image(to_thumbnail(image), use_container_width=True)
        if spec.supports_i2v and len(images) > 1:
            st.caption("This model conditions on a single frame; only the first is used.")

    return images


def gallery(records: list[OutputRecord], columns: int = 3) -> None:
    """Grid of previously rendered clips."""
    if not records:
        st.info("Nothing rendered yet. Generate a clip and it will show up here.")
        return

    for row_start in range(0, len(records), columns):
        row = records[row_start : row_start + columns]
        for column, record in zip(st.columns(columns), row):
            with column:
                st.video(str(record.path))
                st.caption(f"**{record.model}** · {record.created_at}")
                if record.prompt:
                    st.caption(
                        record.prompt[:140] + ("…" if len(record.prompt) > 140 else "")
                    )
                with st.expander("Settings"):
                    st.json(record.metadata, expanded=False)
