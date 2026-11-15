"""Shared small starts for the cell engine's tests: an even start, cell ids, box energies, wrapping checks."""
from __future__ import annotations

import numpy as np

from physics.potts import ENERGY_VOLUME, JOINT, new_universe, touch_pairs


def even_universe(cells_per_side: int = 7, boxes_per_cell_side: int = 4):
    """Energy volumes wrapped in joints with every box at the mean energy (unevenness 0)."""
    return new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0)


def cell_id(cells_per_side: int, cx: int, cy: int, cz: int) -> int:
    """The id new_universe gives cell (cx, cy, cz)."""
    return (cx * cells_per_side + cy) * cells_per_side + cz


def box_energy(universe) -> np.ndarray:
    """Every box's energy, |psi|**2, on the host (float64)."""
    psi = universe.psi.numpy()
    return (psi ** 2).sum(axis=-1)


def set_box_energy(universe, energy: np.ndarray) -> None:
    """Give every box the energy asked, keeping each box's phase (phase 0 where psi was 0)."""
    psi = universe.psi.numpy()
    turn = np.arctan2(psi[..., 1], psi[..., 0])
    amplitude = np.sqrt(np.maximum(energy, 0.0))
    universe.psi.assign(np.stack([amplitude * np.cos(turn), amplitude * np.sin(turn)], axis=-1))


def move_energy(universe, source: int, target: int, keep: float) -> None:
    """Leave `source` cell `keep` of its energy and spread the rest evenly over `target`'s boxes, exactly in sum."""
    owner = universe.owner.numpy()
    energy = box_energy(universe)
    moved = energy[owner == source].sum() * (1.0 - keep)
    energy[owner == source] *= keep
    energy[owner == target] += moved / np.sum(owner == target)
    set_box_energy(universe, energy)


def connected_everywhere(owner: np.ndarray, cells: np.ndarray) -> bool:
    """Whether every listed cell's boxes form one face-connected piece."""
    from scipy import ndimage
    return all(ndimage.label(owner == cell)[1] == 1 for cell in cells)


def wrapped_everywhere(owner: np.ndarray, kind: np.ndarray) -> bool:
    """Whether no energy volume has another energy volume's box, or an empty box, among its boxes' 26 neighbours."""
    low, high = touch_pairs(owner)
    a_high = kind[high] == ENERGY_VOLUME
    a_low = (low >= 0) & (kind[np.maximum(low, 0)] == ENERGY_VOLUME)
    return not np.any(a_high & (a_low | (low < 0)))


def joint_web_pieces(owner: np.ndarray, kind: np.ndarray) -> int:
    """How many face-connected pieces all joint boxes together make."""
    from scipy import ndimage
    joints = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    return int(ndimage.label(joints)[1])
