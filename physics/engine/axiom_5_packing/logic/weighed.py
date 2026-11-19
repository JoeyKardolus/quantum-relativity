"""Weighed: where a staged region's energy is, read against the same ball left alone.

The same start gives the same run, so a ball with a region made fuller and the ball without
it differ only by what the staging did. `extra` is that difference volume by volume;
`middle_of` is where it lies, in volumes from the first volume, and `spread_of` how far it
lies from its own middle. Read on the host, for checks and titles; no step reads it.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store


def volumes_held(store: Store) -> np.ndarray:
    """The plus energy at every volume's place, in calm volumes, as a block of volumes."""
    held = store.cell_quanta.numpy().reshape((store.side,) * 3)[::2, ::2, ::2]
    return held.clip(0) / (QUANTA_A_BOX * store.across ** 3)


def extra(staged: Store, plain: Store) -> np.ndarray:
    """What each volume of the staged ball holds over the same volume of the plain one."""
    return volumes_held(staged) - volumes_held(plain)


def middle_of(more: np.ndarray, part=np.s_[:]) -> np.ndarray:
    """The middle of what is over (what is under counts 0), in volumes from the ball's middle;
    `part` takes one side of the block alone."""
    over, steps = more.clip(0), np.indices(more.shape) - more.shape[0] // 2
    return np.array([(one[part] * over[part]).sum() for one in steps]) / over[part].sum()


def spread_of(more: np.ndarray) -> float:
    """How far what is over lies from its own middle, in volumes (root of the mean square)."""
    over, steps = more.clip(0), np.indices(more.shape) - more.shape[0] // 2
    off = sum((one - middle) ** 2 for one, middle in zip(steps, middle_of(more)))
    return float(np.sqrt((off * over).sum() / over.sum()))
