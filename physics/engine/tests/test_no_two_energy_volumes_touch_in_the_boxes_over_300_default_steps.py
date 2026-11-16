"""Between two energy volumes there is always a joint box, on faces, edges and corners, at every step.

Owner, 2026-11-16: ABAB holds in the Planck boxes, not only in the lattice indices. The
default universe (seed 1, every rule on, splits on) runs 300 steps; after every step
no plus box has a plus box of another energy volume among its 26 neighbours (counted on
the GPU), and every volume stays wrapped (no empty box or wall beside a volume box).
"""
import warp as wp

from physics.engine import Settings, new_universe, stats, step
from physics.engine.inside import inside
from physics.engine.tests._gpu import needs_gpu


@wp.kernel
def _count(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), side: int,
           pairs: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] <= 0:
        return
    here = wp.vec3i(ti, tj, tk)
    for index in range(27):
        there = here + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
        if index != 13 and inside(there, ti // side, side):
            if quanta[there[0], there[1], there[2]] > 0 and owner[there[0], there[1], there[2]] != owner[ti, tj, tk]:
                wp.atomic_add(pairs, 0, 1)


@needs_gpu
def test_no_box_of_one_volume_touches_a_box_of_another_over_300_steps():
    settings = Settings()
    state = new_universe(seed=1)
    pairs = wp.zeros(1, dtype=wp.int32, device=state.quanta.device)
    worst = 0
    for _ in range(300):
        step(state, settings)
        pairs.zero_()
        wp.launch(_count, dim=state.shape, device=state.quanta.device,
                  inputs=[state.quanta, state.owner, state.side, pairs])
        worst = max(worst, int(pairs.numpy()[0]) // 2)
    measured = stats(state, settings)
    print(f"touching volume box pairs, most at any step: {worst}; wrap breaks {measured['wrap_breaks']}; "
          f"splits {measured['splits']}; collapses {measured['collapses']}")
    assert worst == 0 and measured["wrap_breaks"] == 0
