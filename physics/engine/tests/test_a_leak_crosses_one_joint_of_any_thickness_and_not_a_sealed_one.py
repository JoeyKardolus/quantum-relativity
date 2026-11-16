"""Energy leaks from energy volume to energy volume only through the one joint they share, of any thickness, not when sealed.

Owner, 2026-11-16: energy goes volume -> the one shared joint cell -> the volume on the
other side, decided by lattice index, never by guessing from box contents; counted in
Planck boxes a joint is as thick as its energy makes it. Bible rule 4: less the more
squeezed the joint, none past the seal; plus and minus never cancel. A forced cell
lattice (new_universe, 5**3 cells of k**3 boxes, no unevenness, margin 2) with volume A
(lattice 2, 2, 2) at 5000 quanta a box and the other volumes at 1000, and the leak alone
runs 20 rounds of its six passes (leak.py, half a default step each). Joints k = 2 and
4 boxes thick at rest packing (-1000): A's face neighbour B (4, 2, 2) gains plus quanta,
A loses, every joint box keeps exactly its quanta (the leak only reads it). Every joint
at the seal (-2000, squeeze 1): no plus box changes. Then the misleading line: the
boxes of A and B's shared joint on 4 straight x lines through A's face are handed to a
different joint (the edge joint 3, 3, 2) and the leak runs again: A's and B's boxes at
the ends of those lines do not change (the lines no longer run through the joint A and
B share), while their boxes at the ends of the other x lines across that joint do. The
amount and the plus total are exact in every run.
"""
import numpy as np

from physics.engine import Settings, State, host_boxes, leak, new_universe, totals
from physics.engine.tests._gpu import needs_gpu


def _start(thickness: int, joint_quanta: int) -> tuple:
    """Host arrays of the lattice start: quanta, phase, lattice dict, and the place of A's first box."""
    boxes = host_boxes(new_universe(5, thickness, seed=1, unevenness=0.0, margin=2))
    extent = 7
    volume_a = (2 * extent + 2) * extent + 2
    quanta = boxes["quanta"].copy()
    quanta[quanta < 0] = -joint_quanta
    quanta[boxes["owner"] == volume_a] = 5000
    lattice = {"cells_per_side": 5, "owner": boxes["owner"].copy(), "corners": boxes["corners"]}
    return quanta, np.zeros(quanta.shape, dtype=np.float32), lattice


def _run(quanta: np.ndarray, phase: np.ndarray, lattice: dict) -> tuple:
    """The quanta after 20 rounds of the leak alone, the start's and the end's totals."""
    state = State.from_host(quanta, phase, lattice=lattice)
    settings = Settings()
    start = totals(state)
    for round_index in range(20):
        state.step = round_index
        for pass_index, (axis, group) in enumerate((axis, group) for axis in range(3) for group in range(2)):
            leak(state, settings, axis, group, 0.5 * settings.step_time, pass_index)
    return state.quanta.numpy(), start, totals(state)


@needs_gpu
def test_an_open_joint_leaks_at_any_thickness_a_sealed_one_does_not_and_another_joint_carries_nothing():
    extent = 7
    volume_b = (4 * extent + 2) * extent + 2
    for thickness in (2, 4):
        quanta, phase, lattice = _start(thickness, 1000)
        after, start, end = _run(quanta, phase, lattice)
        far = lattice["owner"] == volume_b
        print(f"{thickness} thick: B's plus quanta {int(quanta[far].sum())} -> {int(after[far].sum())}")
        assert after[far].sum() > quanta[far].sum()
        assert np.array_equal(after[quanta < 0], quanta[quanta < 0])
        assert (end["amount"], end["plus"]) == (start["amount"], start["plus"])
    quanta, phase, lattice = _start(2, 2000)
    after, start, end = _run(quanta, phase, lattice)
    print(f"sealed: plus boxes changed {int((after != quanta).sum())}")
    assert np.array_equal(after, quanta)
    assert (end["amount"], end["plus"]) == (start["amount"], start["plus"])
    quanta, phase, lattice = _start(4, 1000)
    owner = lattice["owner"]
    shared = (3 * extent + 2) * extent + 2
    edge = (3 * extent + 3) * extent + 2
    joint_x = np.unique(np.argwhere(owner == shared)[:, 0])
    a_face, b_face = joint_x.min() - 1, joint_x.max() + 1
    lines = [(7, 7), (7, 8), (8, 7), (8, 8)]
    for line_y, line_z in lines:
        owner[joint_x, line_y, line_z] = edge
    after, start, end = _run(quanta, phase, lattice)
    moved_lines = [(line_y, line_z) for line_y in range(6, 10) for line_z in range(6, 10) if (line_y, line_z) not in lines]
    blocked = [after[face, line_y, line_z] == quanta[face, line_y, line_z] for face in (a_face, b_face)
               for line_y, line_z in lines]
    open_ends = [after[a_face, line_y, line_z] != quanta[a_face, line_y, line_z] for line_y, line_z in moved_lines]
    print(f"A's face at x {a_face}, B's at {b_face}; blocked line ends unchanged {sum(blocked)} of {len(blocked)}, "
          f"open line ends changed {sum(open_ends)} of {len(open_ends)}")
    assert all(blocked)
    assert sum(open_ends) > 0
    assert (end["amount"], end["plus"]) == (start["amount"], start["plus"])
