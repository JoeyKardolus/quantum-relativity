"""Joints oscillate: a joint's aim carries it past its balance and back (rule 5); volumes do not swing.

joint_swing is a spring with inertia in each joint's own time. A joint held 30 boxes
over its want, at rest, aims below where it is; letting it follow its aim exactly, it
falls through its want, overshoots below, turns and comes back above: it swings. An
energy volume always aims at its want: its size follows its energy alone.
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, JOINT, Settings, joint_swing
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import even_universe


@needs_gpu
def test_a_joint_overshoots_its_want_and_returns():
    universe = even_universe(4, 4)
    kind = universe.kind.numpy()
    joint = int(np.flatnonzero(kind == JOINT)[0])
    volume = int(np.flatnonzero(kind == ENERGY_VOLUME)[0])
    want = np.full(universe.cell_count, 100.0, dtype=np.float32)
    boxes = np.full(universe.cell_count, 130, dtype=np.int32)
    universe.cell_want.assign(want)
    universe.cell_clock.assign(np.ones(universe.cell_count, dtype=np.float32))
    universe.previous_boxes.assign(boxes.astype(np.float32))
    settings = Settings.all_on()
    sizes = []
    for _ in range(60):
        universe.cell_boxes.assign(boxes)
        joint_swing(universe, settings)
        aim = universe.cell_aim.numpy()
        assert aim[volume] == 100.0
        boxes = np.rint(aim).astype(np.int32)
        sizes.append(int(boxes[joint]))
    print("joint sizes:", sizes[:30])
    first_below = next(index for index, size in enumerate(sizes) if size < 100)
    assert min(sizes) < 90
    assert any(size > 110 for size in sizes[first_below:])
