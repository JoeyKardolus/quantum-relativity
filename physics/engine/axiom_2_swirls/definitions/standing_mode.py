"""Standing mode: the wave along a row of boxes that the step only turns.

One step takes every pair of neighbours in a row once, the even pairs, then the odd.
The slowest uneven wave along the row is solved here from that step itself, so no
number is added: a step leaves it the same wave, every box turned by `turn_per_step`.
"""
import numpy as np

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN


def row_step(across: int) -> np.ndarray:
    """One step of a row of boxes as a matrix: the even pairs, then the odd pairs."""
    spin = np.exp(2j * TURN)
    stays, swaps = (1 + spin) / 2, (1 - spin) / 2  # the pair rotation of two boxes
    step = np.eye(across, dtype=complex)
    for parity in (0, 1):
        sweep = np.eye(across, dtype=complex)
        for low in range(parity, across - 1, 2):
            sweep[low:low + 2, low:low + 2] = [[stays, swaps], [swaps, stays]]
        step = sweep @ step
    return step


def standing_mode(across: int) -> tuple[np.ndarray, float]:
    """The row's standing wave (complex, one number a box) and its turn per step."""
    values, waves = np.linalg.eig(row_step(across))
    sine = np.sin(np.pi * (np.arange(across) - (across - 1) / 2) / across)
    nearest = int(np.argmax(np.abs(waves.conj().T @ sine)))
    mode = waves[:, nearest]
    return mode * np.exp(-1j * np.angle(mode @ sine)), float(np.angle(values[nearest]))
