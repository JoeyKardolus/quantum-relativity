"""Every energy volume stays fully wrapped in joints: 200 steps of the uneven start with every rule on.

Bible foundation 1 (owner, 2026-11-16): an energy volume never touches another energy
volume, not on a face, an edge or a corner, and the joints touch each other in one
web. At the start and every 25 steps, over all 26 neighbours of every box: no box of an
energy volume has a box of another energy volume or an empty box beside it
(stats wrap_breaks 0, and the same recounted here from the boxes); every living cell is
one face-connected piece; and the joint boxes form one connected web.
"""
import numpy as np

from physics.potts import Settings, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import connected_everywhere, joint_web_pieces, wrapped_everywhere


@needs_gpu
def test_no_energy_volume_touches_another_or_empty_space_for_200_steps():
    universe = new_universe(7, 4, seed=1, unevenness=0.3)
    settings = Settings.all_on()
    kind = universe.kind.numpy()
    assert wrapped_everywhere(universe.owner.numpy(), kind)
    assert joint_web_pieces(universe.owner.numpy(), kind) == 1
    for count in range(200):
        step(universe, settings)
        if count % 25 == 24:
            now = stats(universe, settings)
            owner = universe.owner.numpy()
            kind = universe.kind.numpy()
            print(f"step {count + 1}: wrap breaks {now['wrap_breaks']}, web pieces {now['web_pieces']}, "
                  f"collapses {now['collapses']}, splits {now['splits']}, cells {now['cells']}, joints {now['joints']}")
            assert now["wrap_breaks"] == 0
            assert wrapped_everywhere(owner, kind)
            assert connected_everywhere(owner, np.flatnonzero(universe.alive.numpy() == 1))
            assert joint_web_pieces(owner, kind) == 1
