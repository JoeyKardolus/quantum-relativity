"""The picture shows one red glow with a dark hole down its axis, and the tracers as light dots."""
import struct
import zlib

import numpy as np

from physics.engine import one_cell
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.view.logic.paint import paint
from physics.view.logic.save_png import save_png
from physics.view.logic.tracers import lay_tracers


def _down_the_axis(store) -> np.ndarray:
    return paint(store, 0, None, 0.0, 0.0, 2.6, 120, 120).astype(int)


def test_the_picture_has_the_red_volume_and_the_hole():
    picture = _down_the_axis(one_cell())
    red, green, blue = picture[..., 0], picture[..., 1], picture[..., 2]
    assert picture[:5].max() == 0 and picture[:, :5].max() == 0        # black round the volume
    assert red[25:95, 25:95].mean() > 60 and (red >= green).all() and (red >= blue).all()
    ring = np.concatenate([red[60, 30:40], red[60, 80:90], red[30:40, 60], red[80:90, 60]])
    assert red[58:62, 58:62].mean() < 0.4 * ring.mean()                # the open core


def test_the_tracers_are_drawn_as_light_dots_on_the_red():
    store = one_cell()
    with_dots = paint(store, 0, lay_tracers(store, 0, 50), 0.0, 0.0, 2.6, 120, 120).astype(int)
    assert (_down_the_axis(store)[..., 1] < 60).all()                  # the glow alone is red
    assert 20 < (with_dots[..., 1] > 200).sum() <= 50 * 9              # 50 dots, 3 by 3 each


def test_a_saved_picture_reads_back_the_same(tmp_path):
    picture = paint(one_cell(), 0, None, 0.2, 0.15, 2.6, 40, 30)
    save_png(tmp_path / "cell.png", picture)
    data = (tmp_path / "cell.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (40, 30)
    rows = np.frombuffer(zlib.decompress(data[41:-16]), dtype=np.uint8).reshape(30, 121)
    assert np.array_equal(rows[:, 1:].reshape(30, 40, 3), picture)


def test_a_calm_cell_is_drawn_from_the_rule_and_looks_the_same():
    store = one_cell()
    own = _down_the_axis(store)
    assert give_back(store, 0) and store.count == 0
    assert np.array_equal(_down_the_axis(store), own)
