"""Reference-image preparation.

Video models are extremely sensitive to their conditioning frame being the exact
target resolution with the correct aspect ratio. Anything letterboxed or squashed
shows up as drifting geometry across the clip, so we center-crop to the target
aspect ratio ("cover") before resizing rather than padding.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps

__all__ = ["load_image", "prepare_reference", "to_thumbnail"]


def load_image(source: str | Path | bytes | BinaryIO) -> Image.Image:
    """Open an image from a path, raw bytes, or a file-like object (e.g. Streamlit upload)."""
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    # Honour EXIF orientation, then flatten to RGB so RGBA/P/CMYK inputs behave.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def prepare_reference(
    image: Image.Image,
    width: int,
    height: int,
    fit: str = "cover",
) -> Image.Image:
    """Resize ``image`` to exactly ``width`` x ``height``.

    ``cover``   center-crops to the target aspect ratio, then resizes (default).
    ``contain`` fits the whole image inside the frame, padding with black.
    ``stretch`` ignores aspect ratio entirely.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if fit == "stretch":
        return image.resize((width, height), Image.LANCZOS)

    if fit == "contain":
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        fitted = ImageOps.contain(image, (width, height), Image.LANCZOS)
        canvas.paste(
            fitted,
            ((width - fitted.width) // 2, (height - fitted.height) // 2),
        )
        return canvas

    # cover
    return ImageOps.fit(image, (width, height), Image.LANCZOS, centering=(0.5, 0.5))


def to_thumbnail(image: Image.Image, max_side: int = 256) -> Image.Image:
    """A cheap copy for UI previews - never mutate the caller's image."""
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    return thumb
