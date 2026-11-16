"""Joint collapse turns a joint's minus quanta into plus quanta, the same amount, and nowhere else.

Bible rule 4: joint collapse is the one place where pushing energy becomes pulling
energy, the amount kept exactly. Two energy volumes (blocks of +1000 quanta per box)
are separated by a joint slab two boxes thick packed at -4000 per box (squeeze 3, past
the default collapse squeeze 2); everything is wrapped in a joint shell at -1000, whose
outer layer on one side touches empty space and is also packed at -4000, with a small
energy volume (+1000) just inside it, so that layer lies between empty space and plus
quanta. One `collapse` call: every
slab box turns plus with the same amount (+4000), the outer packed layer stays minus (a
joint touching empty space does not collapse), nothing else changes, the plus total
grows by exactly the minus total's loss, and the amount ignoring sign is identical.
"""
import numpy as np

from physics.engine import Settings, State, collapse, totals
from physics.engine.tests._gpu import needs_gpu

SIDE = 24


@needs_gpu
def test_a_squeezed_joint_between_volumes_turns_plus_and_keeps_its_amount():
    quanta = np.zeros((SIDE,) * 3, dtype=np.int32)
    quanta[2:22, 2:22, 2:22] = -1000
    quanta[2, 2:22, 2:22] = -4000
    quanta[3, 10:14, 10:14] = 1000
    quanta[6:11, 6:18, 6:18] = 1000
    quanta[13:18, 6:18, 6:18] = 1000
    quanta[11:13, 6:18, 6:18] = -4000
    state = State.from_host(quanta, np.zeros(quanta.shape))
    before = totals(state)
    collapse(state, Settings())
    after = totals(state)
    result = state.quanta.numpy()
    slab = np.zeros(quanta.shape, dtype=bool)
    slab[11:13, 6:18, 6:18] = True
    print(f"before {before}, after {after}, flipped {int(state.tallies.numpy()[0, 1])}")
    assert np.all(result[slab] == 4000)
    assert np.all(result[~slab] == quanta[~slab])
    assert after["amount"] == before["amount"]
    assert after["plus"] - before["plus"] == before["minus"] - after["minus"] == 4000 * int(slab.sum())
