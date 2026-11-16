"""Every energy volume is wrapped by its own joint, and the joints meet at seams between neighbouring volumes.

Bible foundation: every energy volume is fully wrapped in joints, the joints form one
web, and where joints meet are the seams. A brick start (5**3 cells of 8**3 boxes, an
empty margin of 8; 8 energy volumes in a 2 x 2 x 2 block) is read by cells() and
seams(): 8 energy volumes and 8 joints, each joint wrapping a different volume, each a
face-connected group of minus boxes; every minus box face-touching a volume's box
belongs to a joint (no volume touches empty space); seam faces exist, only between two
joints, and for each of the 12 face-neighbouring volume pairs their joints meet.
"""
import itertools

import numpy as np
from scipy import ndimage

from physics.engine import JOINT, brick_start, cells, seams


def test_eight_volumes_get_eight_joints_that_meet_at_seams():
    quanta, _ = brick_start(5, 8, 8, seed=1)
    found = cells({"quanta": quanta, "clock": np.zeros(quanta.shape, dtype=np.float32)})
    meeting = seams(found)
    volumes = found["volumes"]
    joints = np.flatnonzero(found["kind"] == JOINT)
    print(f"volumes {volumes}, joints {joints.size}, seam faces {meeting['faces']}, "
          f"meeting pairs {len(meeting['pairs'])}, joint sizes {found['boxes'][joints].tolist()}")
    assert volumes == 8 and joints.size == 8
    assert sorted(found["wraps"][joints].tolist()) == list(range(8))
    for joint in joints:
        assert ndimage.label(found["owner"] == joint)[1] == 1
    assert np.all(quanta[meeting["boxes"]] < 0)
    assert meeting["faces"] > 0
    assert np.all(found["kind"][meeting["pairs"]] == JOINT)
    centres = np.rint(found["centre"][:volumes] / 16.0).astype(int)
    met = {tuple(sorted(pair)) for pair in meeting["volumes"].tolist()}
    face_pairs = [(first, second) for first, second in itertools.combinations(range(volumes), 2)
                  if np.abs(centres[first] - centres[second]).sum() == 1]
    assert len(face_pairs) == 12
    assert all(pair in met for pair in face_pairs)
