"""Save PNG: write a picture to a file with nothing but the standard library."""
import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One named piece of a PNG file, with its length before and its check after."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def save_png(path: Path, picture: np.ndarray) -> None:
    """Write (height, width, 3) bytes, top row first, as an RGB picture."""
    rows = b"".join(b"\x00" + row.tobytes() for row in picture)
    head = struct.pack(">IIBBBBB", picture.shape[1], picture.shape[0], 8, 2, 0, 0, 0)
    pieces = _chunk(b"IHDR", head) + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b"")
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + pieces)
