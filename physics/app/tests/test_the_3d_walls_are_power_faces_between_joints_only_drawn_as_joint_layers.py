"""The 3D walls are power-diagram faces, only joint pairs are sent, each as a joint layer whose thickness follows its joint volume."""
import itertools
import json
from types import SimpleNamespace

import numpy as np

from physics.app import Simulation
from physics.app.frame_message import frame_message
from physics.app.joint_volume_shares import joint_volume_shares
from physics.app.power_faces import power_faces
from physics.app.squeeze_shades import squeeze_shades
from physics.app.walls_message import MAX_THICKNESS_SHARE, walls_message

CELLS = 400


def _area(polygon):
    ring = np.asarray(polygon, dtype=float).reshape(-1, 3)
    spokes = ring - ring.mean(axis=0)
    return 0.5 * float(np.linalg.norm(np.cross(spokes, np.roll(spokes, -1, axis=0)).sum(axis=0)))


def test_a_cubic_lattice_gives_unit_square_faces_between_face_neighbours_only():
    points = np.array(list(itertools.product(range(3), repeat=3)), dtype=float)
    every_pair = np.array(list(itertools.combinations(range(len(points)), 2)))
    pairs, polygons = power_faces(points, np.full(len(points), 0.25),
                                  (-0.5, -0.5, -0.5, 2.5, 2.5, 2.5), every_pair)
    steps = np.abs(points[pairs[:, 0]] - points[pairs[:, 1]]).sum(axis=1)
    assert np.allclose(steps, 1.0) and len(pairs) == 54
    assert all(polygon is not None and abs(_area(polygon) - 1.0) < 1e-9 for polygon in polygons)


def test_every_face_corner_has_equal_power_to_both_cells_and_no_lower_power_elsewhere():
    simulation = Simulation(CELLS, seed=2)
    frame = frame_message(simulation)
    positions = np.asarray(frame["positions"]).reshape(-1, 3)
    weights = np.asarray(frame["radius"]) ** 2
    reach = max(frame["radius"])
    box = (*(positions.min(axis=0) - reach), *(positions.max(axis=0) + reach))
    pairs, polygons = power_faces(positions, weights, box, simulation.grid.joints)
    assert any(polygon is not None for polygon in polygons)
    for (first, second), polygon in zip(pairs, polygons):
        if polygon is None:
            continue
        power = ((polygon[:, None, :] - positions[None]) ** 2).sum(axis=2) - weights[None]
        assert np.allclose(power[:, first], power[:, second], atol=1e-6)
        assert (power.min(axis=1) >= power[:, first] - 1e-6).all()


def test_the_walls_message_keeps_the_contract_and_sends_only_joint_pairs():
    simulation = Simulation(CELLS, seed=1)
    simulation.advance()
    frame = frame_message(simulation)
    joints = simulation.grid.joints
    walls = walls_message(np.asarray(frame["positions"]).reshape(-1, 3), frame["radius"], joints,
                          frame["joint_volume"], frame["squeeze"], frame["step"])
    walls = json.loads(json.dumps(walls, allow_nan=False))
    assert set(walls) == {"type", "step", "faces", "mismatch"}
    assert walls["type"] == "walls" and walls["step"] == 1 and 0.0 <= walls["mismatch"] <= 1.0
    joint_of = {tuple(sorted(pair)): index for index, pair in enumerate(joints.tolist())}
    assert walls["faces"]
    for face in walls["faces"]:
        assert set(face) == {"a", "b", "polygon", "thickness", "squeeze"}
        assert len(face["polygon"]) % 3 == 0 and len(face["polygon"]) >= 9
        joint = joint_of[tuple(sorted((face["a"], face["b"])))]
        assert face["squeeze"] == frame["squeeze"][joint]
        ceiling = MAX_THICKNESS_SHARE * min(frame["radius"][face["a"]], frame["radius"][face["b"]])
        assert 0.0 <= face["thickness"] <= ceiling + 1e-4


def test_a_joint_layer_is_thicker_for_a_bigger_joint_volume_and_zero_without_one():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radius, joints = [0.5, 0.5], np.array([[0, 1]])
    thin, thick, none = (walls_message(points, radius, joints, [share], [0.0], 0)["faces"][0]
                         for share in (0.01, 0.02, 0.0))
    assert thick["thickness"] > thin["thickness"] > none["thickness"] == 0.0
    assert walls_message(points, radius, np.zeros((0, 2), dtype=int), [], [], 0)["faces"] == []
    assert walls_message(points, radius, np.zeros((0, 2), dtype=int), [], [], 0)["mismatch"] == 1.0


def test_joint_volume_shares_and_squeeze_shades_read_the_grid():
    grid = SimpleNamespace(joints=np.array([[0, 1], [1, 2]]), volume=np.array([1.0, 3.0, 1.0]),
                           joint_volume=np.array([0.2, 0.5]), squeeze=np.array([0.25, 1.5]))
    assert joint_volume_shares(grid) == [0.1, 0.25]
    assert squeeze_shades(grid) == [0.25, 1.0]
