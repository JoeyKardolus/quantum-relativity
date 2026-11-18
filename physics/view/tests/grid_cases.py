"""Grid cases (test helper): plain-array lattices and a plain-array source, no engine.

`uneven_lattice(side, seed, spread)` returns (kind, length, packing) for a perfect
A B A B lattice whose volumes are about 10 long and joints about 1, each times
exp(spread * a standard normal). `torn_lattice(side)` is the even one after volume 0
tore along y: (kind, length, packing, lines, line_axis). `inside(points, quads)` counts,
for every 2D point, the four-cornered faces (n, 4, 2) it lies strictly inside.
`slice_faces(pieces, layer)` returns the slice's faces (n, 4, 2) with their cell ids.
`turns(faces)` is, per face and corner, how far the outline turns left there (all above
0: the face has area and no fold). `every_layer(source, side)` is a source's layers,
each laid out alone, as one list of `Pieces`.
`FakeSource` fits the source interface: an 8-place lattice whose cells shrink 1% a
step, and that raises at step `fail_at`.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.layout.logic.torn_lines import torn_lines


def uneven_lattice(side: int, seed: int = 1, spread: float = 0.3) -> tuple:
    index = np.indices((side, side, side)).reshape(3, -1)
    kind = np.where((index % 2).sum(axis=0) == 0, 1, -1).astype(np.int8)
    noise = np.exp(spread * np.random.default_rng(seed).standard_normal(side ** 3))
    length = np.where(kind == 1, 10.0, 1.0) * noise
    return kind, length, (np.where(kind == 1, 10.0, 1.0) / length) ** 3


def torn_lattice(side: int) -> tuple:
    kind, length, packing = uneven_lattice(side, spread=0.0)
    kind = np.concatenate([kind, [-1, 1]]).astype(np.int8)
    length = np.concatenate([length, [2.0, 30.0]])
    length[0] = 30.0
    torn_from = np.full(len(kind), -1)
    torn_from[side ** 3:] = 0
    lines, line_axis = torn_lines(side ** 3, kind, torn_from, kind.copy(), np.ones(len(kind), dtype=int))
    return kind, length, np.ones(len(kind)), lines, line_axis


def slice_faces(pieces, layer: int) -> tuple:
    keep = pieces.place[:, 2] == layer
    corners = pieces.corners[keep].astype(np.float64)
    return (0.5 * (corners[:, :4] + corners[:, 4:]))[:, (0, 1, 3, 2), :2], pieces.cell[keep]


def turns(faces: np.ndarray) -> np.ndarray:
    ahead, back = np.roll(faces, -1, axis=1) - faces, faces - np.roll(faces, 1, axis=1)
    return back[..., 0] * ahead[..., 1] - back[..., 1] * ahead[..., 0]


def every_layer(source, side: int) -> list:
    return [source.view(layer) for layer in range(side)]


def inside(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    count = np.zeros(len(points), dtype=np.int64)
    for quad in quads:
        turns = []
        for one in range(4):
            start, end = quad[one], quad[(one + 1) % 4]
            turns.append((end[0] - start[0]) * (points[:, 1] - start[1]) - (end[1] - start[1]) * (points[:, 0] - start[0]))
        turns = np.array(turns)
        count += (turns > 0).all(axis=0) | (turns < 0).all(axis=0)
    return count


class FakeSource:
    SIDE = 8

    def __init__(self, fail_at: int = -1):
        self.fail_at = fail_at
        self.reset()

    def reset(self):
        self.kind, self.length, self.packing = uneven_lattice(self.SIDE, spread=0.1)
        self.steps = 0

    def step(self):
        if self.steps == self.fail_at:
            raise RuntimeError("the source broke")
        self.length = self.length * 0.99
        self.steps += 1

    def view(self, layer=None):
        return cell_pieces(self.SIDE, self.kind, self.length, self.packing, layer=layer)

    def stats(self):
        return {"volumes": int((self.kind == 1).sum()), "joints": int((self.kind == -1).sum()), "size": 1.0}
