"""A saved picture reads back the same."""
import struct
import zlib

import numpy as np

from physics.view.logic.save_png import save_png


def test_a_saved_picture_reads_back_the_same(tmp_path):
    picture = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    save_png(tmp_path / "cell.png", picture)
    data = (tmp_path / "cell.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (40, 30)
    rows = np.frombuffer(zlib.decompress(data[41:-16]), dtype=np.uint8).reshape(30, 121)
    assert np.array_equal(rows[:, 1:].reshape(30, 40, 3), picture)
