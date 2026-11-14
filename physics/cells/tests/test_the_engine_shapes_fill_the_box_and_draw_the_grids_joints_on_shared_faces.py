"""The engine's shapes fill the periodic box exactly and draw the grid's joints on shared faces.

Checked independently of the engine's own sums, on a new engine and after 10
steps at the default settings (500 energy volumes, seed 4, unevenness 0.1):
  fill    every cell's volume, summed here from the faces shapes() returns (a fan
          of tetrahedra around the mean corner), adds up to the box volume within
          0.5%, and each cell's volume matches the tessellation's own.
  joints  pyvoro2's public compute() run afresh on the engine's positions and
          weights gives the cell pairs that share a face, the same set as the
          tessellation's pairs. On the new engine the joints in shapes(), the
          grid's joints and those pairs are one set (0% mismatch); after the steps
          the grid keeps its own (woven) joints and shapes() draws exactly the
          joints that still have a face. Every joint polygon is one of cell a's
          faces, and every pair has as many faces seen from a as from b.
"""
from __future__ import annotations

import numpy as np
import pyvoro2
import pytest

from physics.cells import EngineSettings, new_engine, shapes, step


@pytest.fixture(scope="module")
def engines():
    start = new_engine(500, seed=4, unevenness=0.1)
    later = start
    for _ in range(10):
        later = step(later, EngineSettings.all_on())
    return [start, later]


def _cell_volume(faces):
    polygons = [np.asarray(face).reshape(-1, 3) for face in faces]
    apex = np.concatenate(polygons).mean(axis=0)
    total = 0.0
    for polygon in polygons:
        for second, third in zip(polygon[1:-1], polygon[2:]):
            total += abs(np.dot(np.cross(second - polygon[0], third - polygon[0]),
                                apex - polygon[0])) / 6
    return total


def test_shapes_fill_the_box_exactly(engines):
    for engine in engines:
        drawn = shapes(engine)
        volumes = np.array([_cell_volume(cell["faces"]) for cell in drawn["cells"]])
        assert abs(volumes.sum() / engine.side ** 3 - 1) < 0.005
        assert np.allclose(volumes, engine.tessellation.volume, rtol=1e-6)
        assert len(drawn["cells"]) == engine.grid.cell_count


def test_drawn_joints_are_the_grids_joints_that_share_a_face(engines):
    for engine in engines:
        domain = pyvoro2.OrthorhombicCell(bounds=((0.0, engine.side),) * 3,
                                          periodic=(True, True, True))
        records = pyvoro2.compute(engine.positions, domain=domain, mode="power",
                                  weights=engine.weights, output="cells")
        seen_from = {}
        for record in records:
            for face in record["faces"]:
                other = face["adjacent_cell"]
                if other != record["id"]:
                    key = (min(record["id"], other), max(record["id"], other))
                    seen_from.setdefault(key, []).append(record["id"])
        drawn = shapes(engine)
        drawn_pairs = {(joint["a"], joint["b"]) for joint in drawn["joints"]}
        grid_pairs = set(map(tuple, engine.grid.joints.tolist()))
        assert set(seen_from) == set(map(tuple, engine.tessellation.pairs.tolist()))
        assert drawn_pairs == grid_pairs & set(seen_from)
        if engine is engines[0]:
            assert drawn_pairs == grid_pairs == set(seen_from)
        for (low, high), sides in seen_from.items():
            assert sides.count(low) == sides.count(high)
        for joint in drawn["joints"]:
            assert joint["a"] < joint["b"]
            assert joint["polygon"] in drawn["cells"][joint["a"]]["faces"]
            assert 0 <= joint["push"] <= 1 and joint["thickness"] > 0
        assert all(0 <= cell["pull"] <= 1 for cell in drawn["cells"])
