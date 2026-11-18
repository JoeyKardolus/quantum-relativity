"""PNG chunk: one chunk of a PNG file.

Contract: `png_chunk(name, body)` returns the chunk's bytes as the PNG format writes
them: the body's length (4 bytes, big end first), the four-letter name, the body, and
the CRC-32 of name and body (4 bytes).
"""
from __future__ import annotations

import struct
import zlib


def png_chunk(name: bytes, body: bytes) -> bytes:
    """Length, name, body and checksum of one chunk."""
    return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body))
