"""Two neighbouring energy volumes share exactly one joint, and every red | blue | red line crosses that one joint.

Owner, 2026-11-16: one shared joint between two neighbouring energy volumes (red |
blue | red), not a joint shell per volume; joints touch each other in a connected web
and every energy volume stays fully wrapped. A brick start (5**3 cells of 8**3 boxes,
an empty margin of 8; 8 energy volumes in a 2 x 2 x 2 block) is read by cells() and
seams(): 20 joints, 12 between the 12 face-neighbouring volume pairs and 8 skins
between one volume and empty space; no two joints lie between the same two volumes;
each joint is one face-connected group of minus boxes; the joints form one web and
meet at seams; no volume touches empty space. Every straight line along an axis from a
plus box through minus boxes to a plus box of another volume (the leak's crossing,
leak.py) runs through minus boxes of one joint, the one between those two volumes.
"""
import itertools

import numpy as np
from scipy import ndimage

from physics.engine import JOINT, brick_start, cells, seams
from physics.engine.tests.crossings import crossings


def test_one_joint_between_each_neighbouring_pair_and_every_crossing_goes_through_it():
    quanta, _ = brick_start(5, 8, 8, seed=1)
    found = cells({"quanta": quanta, "clock": np.zeros(quanta.shape, dtype=np.float32)})
    meeting = seams(found)
    volumes = found["volumes"]
    joints = np.flatnonzero(found["kind"] == JOINT)
    between = [tuple(pair) for pair in found["between"][joints].tolist()]
    through, total = crossings(quanta, found)
    print(f"volumes {volumes}, joints {joints.size}, seam faces {meeting['faces']}, crossings {total}, "
          f"through their own joint {through}, joint sizes {found['boxes'][joints].tolist()}")
    assert volumes == 8 and joints.size == 20
    assert len(set(between)) == len(between)
    centres = np.rint(found["centre"][:volumes] / 16.0).astype(int)
    face_pairs = {(first, second) for first, second in itertools.combinations(range(volumes), 2)
                  if np.abs(centres[first] - centres[second]).sum() == 1}
    assert len(face_pairs) == 12
    assert {pair for pair in between if pair[0] >= 0} == face_pairs
    assert sorted(pair[1] for pair in between if pair[0] < 0) == list(range(volumes))
    for joint in joints:
        assert ndimage.label(found["owner"] == joint)[1] == 1
    assert ndimage.label(quanta < 0)[1] == 1
    assert meeting["faces"] > 0 and np.all(quanta[meeting["boxes"]] < 0)
    assert total > 0 and through == total
