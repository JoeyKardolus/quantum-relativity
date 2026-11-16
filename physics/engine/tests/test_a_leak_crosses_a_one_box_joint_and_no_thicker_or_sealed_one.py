"""Energy leaks from energy volume to energy volume across a one-box joint, never into it, and not when thick or sealed.

Bible rule 4: energy leaks from energy volume to energy volume only through joints (A
to B to A), less the more squeezed the joint, none past the seal; plus and minus never
cancel. Two energy volumes (blocks of plus boxes, A at 5000 quanta a box, B at 1000)
sit inside a joint shell with a joint slab between them, and the leak alone runs 20
rounds of its six passes (leak.py, half a default step each). Slab one box thick at
rest packing (-1000): B gains plus quanta, A loses as many, every joint box keeps
exactly its quanta (the leak only reads it). Slab two boxes thick: B's plus quanta do
not change. Slab one box thick with every joint at the seal (-2000, squeeze 1): B's
plus quanta do not change. The amount and the plus total are exact in all three.
"""
import numpy as np

from physics.engine import Settings, State, leak, totals
from physics.engine.tests._gpu import needs_gpu

SIDE = 24


def _run(thickness: int, joint_quanta: int) -> tuple:
    """B's plus quanta before and after the leak, the joint boxes before and after, and the amounts."""
    quanta = np.zeros((SIDE,) * 3, dtype=np.int32)
    quanta[2:22, 2:22, 2:22] = -joint_quanta
    quanta[5:11, 6:18, 6:18] = 5000
    quanta[11 + thickness:17 + thickness, 6:18, 6:18] = 1000
    far = np.zeros(quanta.shape, dtype=bool)
    far[11 + thickness:17 + thickness, 6:18, 6:18] = True
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32))
    settings = Settings()
    start = totals(state)
    for round_index in range(20):
        state.step = round_index
        for pass_index, (axis, group) in enumerate((axis, group) for axis in range(3) for group in range(2)):
            leak(state, settings, axis, group, 0.5 * settings.step_time, pass_index)
    result = state.quanta.numpy()
    return int(quanta[far].sum()), int(result[far].sum()), quanta[quanta < 0], result[quanta < 0], start, totals(state)


@needs_gpu
def test_a_thin_open_joint_leaks_and_a_thick_or_sealed_one_does_not():
    thin = _run(1, 1000)
    thick = _run(2, 1000)
    sealed = _run(1, 2000)
    print(f"B's plus quanta: thin {thin[0]} -> {thin[1]}, thick {thick[0]} -> {thick[1]}, "
          f"sealed {sealed[0]} -> {sealed[1]}")
    assert thin[1] > thin[0]
    assert np.array_equal(thin[2], thin[3])
    assert thick[1] == thick[0]
    assert sealed[1] == sealed[0]
    for run in (thin, thick, sealed):
        assert run[5]["amount"] == run[4]["amount"]
        assert run[5]["plus"] == run[4]["plus"]
