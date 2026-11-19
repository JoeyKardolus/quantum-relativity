"""Standing mode: the whirl round a closed ring of boxes that the step only turns.

A ring whirl's boxes go round its empty core and close on themselves. One step takes every
pair of neighbours round the ring: the even pairs by half, the odd pairs, the even pairs
by the other half, so a step is the same forward and backward. The wave that winds one
whole turn round the ring is solved here from that step itself, so no number is added: a
step leaves it the same wave, every box turned by `turn_per_step`. Every box holds the
same, and its phase falls from box to box the way the fluid goes.
"""
import numpy as np

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN

SWEEPS = ((0, TURN / 2), (1, TURN), (0, TURN / 2))  # round the core: parity and turn


def ring_step(across: int) -> np.ndarray:
    """One step of a closed ring of boxes as a matrix: half the even pairs, the odd, the other half."""
    step = np.eye(across, dtype=complex)
    for parity, turn in SWEEPS:
        spin = np.exp(2j * turn)
        stays, swaps = (1 + spin) / 2, (1 - spin) / 2  # the pair rotation of two boxes
        sweep = np.zeros((across, across), dtype=complex)
        for low in range(parity, across, 2):
            high = (low + 1) % across
            sweep[[low, high], [low, high]], sweep[[low, high], [high, low]] = stays, swaps
        step = sweep @ step
    return step


def standing_mode(across: int) -> tuple[np.ndarray, float]:
    """The ring's standing whirl of one whole turn (complex, one number a box) and its turn per step."""
    falls = np.exp(-2j * np.pi * np.arange(across) / across)  # one whole turn, the phase falling
    halves = np.stack([falls * (np.arange(across) % 2 == odd) for odd in (0, 1)], axis=1)
    values, mixes = np.linalg.eig(halves.conj().T @ ring_step(across) @ halves)
    nearest = int(np.argmax(np.abs(mixes.sum(axis=0)) / np.linalg.norm(mixes, axis=0)))
    mode = halves @ mixes[:, nearest]
    return mode / mode[0], float(np.angle(values[nearest]))
