"""Axiom 04: joints form one web; two neighbouring volumes share exactly one joint; energy crosses volume -> joint -> volume.

Bible "The foundation" 1 and 4. Default universe (seed 1, Settings(), axiom_run.py), 300
steps. Every 10 steps: all minus boxes form one face-connected piece, and every pair of
face-neighbouring energy volumes (lattice places 2 apart on one axis) has exactly one
joint between them, holding at least one box. Every step: energy goes volume -> joint
-> volume, never volume -> volume. Plus quanta never take a joint's minus sign (axiom
02), so they cannot sit in a joint; they cross it in one handover between the plus
boxes on either side (bible, "How the engine holds the foam"), and the engine books
every crossing on the joint it went through (state.crossed: quanta from between[0] to
between[1]). In a step without collapse or split, every volume's change of quanta must
equal what its own joints booked into and out of it; quanta that a volume gained or
lost beyond that went some other way, straight from volume to volume. Those quanta,
summed over the run, must be 0, and crossings must have happened (the rule runs).
(Test changed 2026-11-16 from "no volume's quanta ever rise in a quiet step": with
plus quanta never turning minus, that could only hold if no energy ever crossed, which
axiom 09 requires.)
"""
import numpy as np

from physics.engine.tests._gpu import needs_gpu


def _pairs(found: dict) -> list:
    """Face-neighbouring energy volume pairs (cells() ids), from their lattice places."""
    places = found["place"][:found["volumes"]]
    gaps = np.abs(places[:, None, :] - places[None, :, :])
    face = (gaps.sum(axis=2) == 2) & (gaps.max(axis=2) == 2)
    return [(int(first), int(second)) for first, second in zip(*np.nonzero(np.triu(face)))]


@needs_gpu
def test_the_joints_are_one_web_with_one_joint_per_pair_and_energy_never_skips_the_joint():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    pieces = [check["web_pieces"] for check in run["checks"]]
    bad_pairs = 0
    pair_count = 0
    for check in run["checks"]:
        found = check["cells"]
        between = [tuple(int(value) for value in row) for row in found["between"]]
        for pair in _pairs(found):
            pair_count += 1
            joints = [index for index, row in enumerate(between) if row == pair]
            if len(joints) != 1 or found["boxes"][joints[0]] == 0:
                bad_pairs += 1
    direct, booked, quiet_steps = 0, 0, 0
    previous = run["cell_amount0"]
    previous_collapses, previous_splits = 0, 0
    for row in run["steps"]:
        quiet = row["collapses"] == previous_collapses and row["splits"] == previous_splits
        if quiet and row["cell_amount"].shape == previous.shape:
            quiet_steps += 1
            expected = np.zeros(previous.shape, dtype=np.int64)
            crossed = row["crossed"]
            joints = np.flatnonzero(crossed)
            np.add.at(expected, row["between"][joints, 0], -crossed[joints])
            np.add.at(expected, row["between"][joints, 1], crossed[joints])
            volumes = np.flatnonzero(run["kinds"] == 1)
            change = row["cell_amount"][volumes] - previous[volumes]
            direct += int(np.abs(change - expected[volumes]).sum())
            booked += int(np.abs(crossed).sum())
        previous, previous_collapses, previous_splits = row["cell_amount"], row["collapses"], row["splits"]
    print(f"web pieces over checks: min {min(pieces)}, max {max(pieces)}; face-neighbour volume pairs checked "
          f"{pair_count}, without exactly one joint {bad_pairs}; quiet steps {quiet_steps}; quanta booked "
          f"through joints {booked}; quanta a volume gained or lost beyond its joints' bookings: {direct}")
    assert max(pieces) == 1 and bad_pairs == 0
    assert direct == 0 and booked > 0
