"""Pull: one step of every volume being pulled toward every other, by what the other held a delay ago.

On the host, all pairs (`definitions/pull.py` has the law and what it costs). Each step the
volumes' energy is written into the past kept (`cell_past`, a ring of `REACH` steps); a
collapsed joint's plus counts half to each of the two volumes it merged. Then for each pair:
the way between (`between`), how late the pull is (`steps_late`), the other's energy then.
What comes of it is written twice: how deep each volume lies (`cell_pull`), and its fluid's
flow, which gains its quanta times the pull on it times the step, in whole numbers, never
past the top speed. A volume is in no sum of its own. Before a pull has had the time to
arrive there is none.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_5_packing.definitions.pull import REACH, steps_late
from physics.engine.axiom_5_packing.logic.between import between, held_of, volumes_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED


def _energy(held: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Each volume's pulling energy: its own plus, and half of each collapsed joint's at its sides."""
    plus = np.pad(held.clip(0), 1)
    sides = sum(plus[tuple((at + 1 + way * np.eye(3, dtype=int)[axis]).T)] for axis in range(3) for way in (-1, 1))
    return plus[tuple((at + 1).T)] + sides / 2


def pulls(store: Store) -> tuple:
    """The own volumes, how deep each lies and the pull on each along the three directions (a speed a turn)."""
    held, a_turn = held_of(store), store.turn_per_step / (2.0 * np.pi)
    cells = volumes_of(store, held)
    at = steps_of(cells, store.side)
    if store.cell_past is None:
        store.cell_past = wp.zeros((REACH, store.side ** 3), dtype=wp.float32, device="cpu")
    past, energy = store.cell_past.numpy(), _energy(held, at)
    past[store.step % REACH] = 0.0
    past[store.step % REACH, cells] = energy
    way = between(held, at)
    apart = np.linalg.norm(way, axis=2)
    np.fill_diagonal(apart, np.inf)  # no volume is another to itself
    late = steps_late(apart, a_turn)
    then = np.where(late <= store.step, past[(store.step - late) % REACH, cells[None, :]], 0.0)
    return cells, energy + (then / apart).sum(axis=1), ((then / apart ** 3)[..., None] * way).sum(axis=1)


def pull(store: Store) -> None:
    """Write how deep every own volume lies and add a step of the pull on it to its flow."""
    if store.side == 0:
        return
    cells, deep, drawn = pulls(store)
    depth, flow = np.zeros(store.side ** 3, dtype=np.float32), store.cell_flow.numpy()
    depth[cells] = deep
    quanta = store.cell_quanta.numpy()[cells].astype(np.float64)[:, None]
    gain = np.floor(quanta * drawn * store.turn_per_step / (2.0 * np.pi) + 0.5).astype(np.int64)
    most = (quanta * TOP_SPEED).astype(np.int64)  # nothing goes faster than the top speed
    flow[cells] = np.clip(flow[cells] + gain, -most, most)
    store.cell_pull.assign(depth)
    store.cell_flow.assign(flow)
