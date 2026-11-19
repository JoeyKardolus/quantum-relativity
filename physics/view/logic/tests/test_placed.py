"""The picture's distances are the engine's: joined volumes are drawn as far apart as `apart` says."""
import numpy as np

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.view.logic.placed import placed, settle
from physics.view.logic.title_line import squeeze_read

NEAR = 0.25  # a joint's drawn length may be this share off the engine's; the mean far less


def _grown(steps: int) -> tuple:
    """A cluster grown `steps` steps, its middles settled every step as the window does."""
    store, middle = first_volume(layers=4), np.zeros((0, 3))
    for _ in range(steps):
        step(store)
        middle = settle(store, middle)[0]
    return store, *settle(store, middle)


def test_centre_and_rim_joints_are_drawn_at_the_engine_s_distance_and_volumes_at_one_scale():
    store, middle, size, joints, volumes, apart = _grown(190)
    cell, before, after, _axis = joints.T
    drawn = np.linalg.norm(middle[after] - middle[before], axis=1) / apart
    out = np.abs(steps_of(cell, store.side) - store.side // 2).max(axis=1)
    centre, rim = out <= 2, out >= out.max() - 1
    assert out.max() >= 5 and centre.sum() > 20 and rim.sum() > 20  # three layers stand
    assert np.median(apart[centre]) < 0.85 * np.median(apart[rim])  # the engine: closer at the centre
    for part in (centre, rim):
        assert np.abs(drawn[part] - 1).max() < NEAR and abs(np.median(drawn[part]) - 1) < 0.1
    assert np.sqrt(np.mean((drawn - 1) ** 2)) < 0.08  # what the title calls "drawn off"
    assert "red to red: centre" in squeeze_read(store, middle) and "drawn off" in squeeze_read(store, middle)
    shapes = placed(store, middle, size, joints, 7.0).numpy()
    wide = shapes[:, 1, 0][shapes[:, 2, 2] > 0] / size[volumes]
    assert np.allclose(wide, 1 / 7.0, rtol=1e-5)  # every volume as wide as its size, one scale


def test_a_joint_s_block_leans_from_the_one_of_its_volumes_to_the_other():
    store, middle, size, joints, _volumes, _apart = _grown(150)
    shapes = placed(store, middle, size, joints, 1.0).numpy()
    slot = np.full(len(size), -1)
    slot[store.own_cell.numpy()[:store.count]] = np.arange(store.count)
    cell, before, after, axis = joints.T
    lean = np.zeros((len(cell), 3))
    np.put_along_axis(lean, np.array([[1, 2], [0, 2], [0, 1]])[axis], shapes[slot[cell], 2, :2], 1)
    lean[np.arange(len(cell)), axis] = 1.0
    assert np.allclose(lean * shapes[slot[cell], 1, :1], middle[after] - middle[before], atol=1e-3)
    assert np.allclose(shapes[slot[cell], 2, 2], -1.0 - axis)
