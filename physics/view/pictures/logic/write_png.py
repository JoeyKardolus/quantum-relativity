"""Write PNG: an RGB image to a PNG file, with the standard library only.

Contract: `write_png(path, picture)` writes uint8 (height, width, 3) RGB, row 0 at the
top, as an 8-bit truecolour PNG and returns the number of bytes written. Raises
ValueError for any other shape.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from physics.view.pictures.definitions.picture_marks import PNG_SIGNATURE
from physics.view.pictures.definitions.png_chunk import png_chunk


def write_png(path, picture: np.ndarray) -> int:
    """Save an RGB image as a PNG file."""
    picture = np.ascontiguousarray(picture, dtype=np.uint8)
    if picture.ndim != 3 or picture.shape[2] != 3:
        raise ValueError(f"a picture is (height, width, 3), got {picture.shape}")
    height, width = picture.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), picture.reshape(height, width * 3)], axis=1)
    data = (PNG_SIGNATURE + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 9)) + png_chunk(b"IEND", b""))
    Path(path).write_bytes(data)
    return len(data)
