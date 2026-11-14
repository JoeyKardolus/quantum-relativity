"""The slice's foam cells tile the view box, a bigger radius owns a bigger cell, and the message keeps the contract."""
import json

import numpy as np
from fastapi.testclient import TestClient

from physics.app import Simulation, create_app
from physics.app.frame_message import frame_message
from physics.app.power_cells import power_cells
from physics.app.slice_message import slice_message

CELLS = 400


def _area(flat):
    ring = np.asarray(flat, dtype=float).reshape(-1, 2)
    return 0.5 * abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
                           - np.dot(ring[:, 1], np.roll(ring[:, 0], -1))))


def _even_slice(depth, axis):
    simulation = Simulation(CELLS, seed=3, unevenness=0.0)
    frame = frame_message(simulation)
    positions = np.asarray(frame["positions"]).reshape(-1, 3)
    cut = slice_message(positions, frame["radius"], frame["energy"], simulation.grid.joints,
                        frame["squeeze"], depth, axis, frame["step"])
    return positions, cut


def test_the_slice_of_an_even_grid_tiles_the_view_box_without_gap_or_overlap():
    for axis, plane_axes in (("x", [1, 2]), ("y", [0, 2]), ("z", [0, 1])):
        for depth in (0.2, 0.5, 0.8):
            positions, cut = _even_slice(depth, axis)
            flat = positions[:, plane_axes]
            box_area = float(np.prod(flat.max(axis=0) - flat.min(axis=0)))
            areas = [_area(cell["polygon"]) for cell in cut["cells"]]
            assert len(areas) > 10 and min(areas) > 0
            assert abs(sum(areas) / box_area - 1.0) < 0.01, (axis, depth)


def test_a_square_lattice_gives_square_cells_and_a_bigger_radius_a_bigger_cell():
    grid = np.array([[col, row] for row in range(5) for col in range(5)], dtype=float)
    box = (-0.5, -0.5, 4.5, 4.5)
    even, walls = power_cells(grid, np.full(25, 0.25), box)
    assert all(abs(_area(polygon) - 1.0) < 1e-9 for polygon in even)
    assert len(walls) == 40
    weights = np.full(25, 0.25)
    weights[12] = 0.6
    grown, _ = power_cells(grid, weights, box)
    assert _area(grown[12]) > 1.2 and abs(sum(map(_area, grown)) - 25.0) < 1e-9


def test_the_slice_message_is_plain_json_with_the_contract_keys_and_joint_walls():
    simulation = Simulation(CELLS, seed=1)
    simulation.advance()
    frame = frame_message(simulation)
    positions = np.asarray(frame["positions"]).reshape(-1, 3)
    cut = slice_message(positions, frame["radius"], frame["energy"], simulation.grid.joints,
                        frame["squeeze"], 0.5, "y", frame["step"])
    cut = json.loads(json.dumps(cut, allow_nan=False))
    assert set(cut) == {"type", "step", "depth", "cells", "walls"}
    assert cut["type"] == "slice" and cut["step"] == 1 and cut["depth"] == 0.5
    assert all(set(cell) == {"id", "polygon", "energy"} and len(cell["polygon"]) % 2 == 0
               and all(round(value, 3) == value for value in cell["polygon"])
               for cell in cut["cells"])
    joints = {tuple(sorted(pair)): index for index, pair in enumerate(simulation.grid.joints.tolist())}
    for wall in cut["walls"]:
        assert set(wall) == {"a", "b", "line", "squeeze"} and len(wall["line"]) == 4
        joint = joints.get(tuple(sorted((wall["a"], wall["b"]))))
        assert wall["squeeze"] == (None if joint is None else frame["squeeze"][joint])
    assert any(wall["squeeze"] is not None for wall in cut["walls"])


def test_the_socket_sends_a_slice_after_each_frame_and_moves_the_cut_on_request():
    with TestClient(create_app(CELLS, seed=1)) as client:
        with client.websocket_connect("/ws") as socket:
            greeting = [socket.receive_json() for _ in range(4)]
            assert [message["type"] for message in greeting] == ["settings", "mesh", "frame", "slice"]
            assert "walls_not_matching_joints" in greeting[2]["stats"]
            socket.send_json({"type": "slice", "depth": 2.0, "axis": "x"})
            moved = socket.receive_json()
            assert moved["type"] == "slice" and moved["depth"] == 1.0
