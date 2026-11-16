"""Every energy volume has joints on all 26 sides, and two face-neighbouring volumes share exactly one joint.

Owner, 2026-11-16: the cells form a forced grid; a cell is an energy volume when all
three lattice indices are even, otherwise a joint, so every volume is wrapped in joints
on all 26 sides (faces, edges, corners), the joints form one web and two neighbouring
volumes share exactly one joint cell (red | blue | red), fixed by index forever. For
lattices of 3 to 9 cells a side (lattice_kinds): every volume's 26 lattice neighbours
are joints, and every joint cell lies between at most one pair of volumes. The default
start shape (new_universe, seed 1, 5**3 cells of 8**3 boxes, margin 8) read by cells():
8 volumes, 12 face-neighbouring pairs, each with exactly one joint whose `between` is
that pair; every plus box's 26 neighbouring boxes are its own volume's or joints'; the
joint boxes form one face-connected web; and every straight line from a plus box
through minus boxes to a plus box of another volume runs through the one joint those
two volumes share (crossings.py), 100% of them. After 100 default steps the corners
have moved and straight lines also cross other joints (the share is printed); the leak
then runs its six passes alone and every plus box it changed ends a line through the
one joint its two volumes share (through_ends.py): 100% of the leak's crossings.
"""
import itertools

import numpy as np
from scipy import ndimage

from physics.engine import JOINT, Settings, cells, host_boxes, lattice_kinds, leak, new_universe, step
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.crossings import crossings
from physics.engine.tests.through_ends import through_ends


def test_every_volume_of_any_lattice_has_26_joint_neighbours_and_one_joint_per_face_pair():
    for cells_per_side in range(3, 10):
        kinds = lattice_kinds(cells_per_side)
        volumes = np.argwhere(kinds == 1)
        assert volumes.size > 0
        for volume in volumes:
            for step in itertools.product((-1, 0, 1), repeat=3):
                if step != (0, 0, 0):
                    assert kinds[tuple(volume + np.asarray(step))] == -1
        for joint in np.argwhere(kinds == -1):
            pairs = 0
            for axis in range(3):
                up, down = joint.copy(), joint.copy()
                up[axis] += 1
                down[axis] -= 1
                if kinds[tuple(up)] == 1 and kinds[tuple(down)] == 1:
                    pairs += 1
            assert pairs <= 1


@needs_gpu
def test_each_face_pair_of_volumes_shares_one_joint_and_every_crossing_runs_through_it():
    state = new_universe(5, 8, seed=1, margin=8)
    boxes = host_boxes(state)
    quanta = boxes["quanta"]
    found = cells(boxes)
    volumes = found["volumes"]
    joints = np.flatnonzero(found["kind"] == JOINT)
    between = [tuple(pair) for pair in found["between"][joints].tolist() if pair[0] >= 0]
    place = found["place"]
    face_pairs = {(first, second) for first, second in itertools.combinations(range(volumes), 2)
                  if np.abs(place[first] - place[second]).sum() == 2}
    through, total = crossings(quanta, found)
    print(f"volumes {volumes}, joints {joints.size}, face pairs {len(face_pairs)}, crossings {total}, "
          f"through their own joint {through}")
    assert volumes == 8 and len(face_pairs) == 12
    assert sorted(between) == sorted(face_pairs)
    owner = found["owner"]
    padded = np.pad(owner, 1, constant_values=-1)
    padded_quanta = np.pad(quanta, 1)
    side = quanta.shape[0]
    for step in itertools.product(range(3), repeat=3):
        if step != (1, 1, 1):
            near = padded[step[0]:step[0] + side, step[1]:step[1] + side, step[2]:step[2] + side]
            near_quanta = padded_quanta[step[0]:step[0] + side, step[1]:step[1] + side, step[2]:step[2] + side]
            plus = quanta > 0
            assert np.all((near[plus] == owner[plus]) | (near_quanta[plus] < 0))
    assert ndimage.label(quanta < 0)[1] == 1
    assert total > 0 and through == total


@needs_gpu
def test_after_100_steps_the_leak_changes_only_boxes_on_lines_through_their_shared_joint():
    state = new_universe(5, 8, seed=1, margin=8)
    settings = Settings()
    for _ in range(100):
        step(state, settings)
    boxes = host_boxes(state)
    found = cells(boxes)
    before = boxes["quanta"]
    through, total = crossings(before, found)
    for pass_index, (axis, group) in enumerate((axis, group) for axis in range(3) for group in range(2)):
        leak(state, settings, axis, group, 0.5 * settings.step_time, pass_index)
    after = state.quanta.numpy()
    changed = after != before
    allowed = through_ends(before, found)
    print(f"after 100 steps: straight lines between two volumes {total}, through their own joint {through}; "
          f"boxes the leak changed {int(changed.sum())}, on such lines {int((changed & allowed).sum())}")
    assert changed.sum() > 0
    assert not np.any(changed & ~allowed)
