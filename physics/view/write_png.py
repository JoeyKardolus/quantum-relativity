"""Write PNG: an RGB image to a PNG file, with the standard library only.

Contract: `write_png(path, picture)` writes uint8 (height, width, 3) RGB, row 0 at the
top, as an 8-bit truecolour PNG and returns the number of bytes written.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path, picture: np.ndarray) -> int:
    """Save an RGB image as a PNG file."""
    picture = np.ascontiguousarray(picture, dtype=np.uint8)
    height, width = picture.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), picture.reshape(height, width * 3)], axis=1)
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) \
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 9)) + _chunk(b"IEND", b"")
    Path(path).write_bytes(data)
    return len(data)


def _chunk(name: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body))
