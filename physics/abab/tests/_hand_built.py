"""A hand-built ABAB structure for the tests: energy volumes and joints with chosen contacts."""
import numpy as np

from physics.abab import KIND_A, State


def hand_built(kind, energy, size, contacts, rate=None) -> State:
    """A State from plain lists; start sizes from these sizes, mean over the energy volumes."""
    kind = np.array(kind, dtype=np.uint8)
    size = np.array(size, dtype=float)
    return State(kind=kind, energy=np.array(energy, dtype=float), size=size,
                 rate=np.zeros(len(kind)) if rate is None else np.array(rate, dtype=float),
                 contacts=np.unique(np.array(contacts, dtype=np.int64), axis=0), collapses=0, splits=0,
                 start_size=float(size.sum()), start_mean_size=float(size[kind == KIND_A].mean()))
