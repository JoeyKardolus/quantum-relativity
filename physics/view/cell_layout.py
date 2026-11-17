"""Cell layout: the corners of the warped grid, from how wide every slot is.

Contract: `cell_layout(widths)` takes (3, side, side, side) float widths, widths[axis]
[i, j, k] the width along `axis` of the slot at lattice place (i, j, k) (`cell_slots`),
and returns float64 (side + 1, side + 1, side + 1, 3): the grid's corner points,
centred on 0. The slot at (i, j, k) is the block between corners [i, j, k] and
[i + 1, j + 1, k + 1], so slots share their corners: they leave no gap and keep their
places in the pattern whatever their sizes.
A cell has a size and no shape, and a row of big cells beside a row of small ones
cannot both be drawn exactly their size in one grid without gaps (summed along 16
volumes, rows of the default start differ by several joints' thickness). The corners
are therefore the best fit: every edge of every slot wants to be as long as the slot
is wide along it and to point straight along its axis, and the corners settle where
the squared misses, each divided by the length wanted, are least in sum (a miss
sideways counts SHEAR times a miss in length). Dividing by the length makes a thin
joint as stiff as it is thin, so mortar stays mortar and the big energy volumes take
up the mismatch. Where all sizes agree the fit is exact. Where they do not, a region
of small cells is drawn small, but one cell smaller than all its neighbours is drawn
only part of the way smaller (measured on the default start, 2026-11-17: drawn size
follows a volume's own length to the power 0.46, off by 5% on average, 27% at most;
no cell folded there or at three times the unevenness). The colour carries each
cell's own packing exactly.
The fit is one sparse linear solve per coordinate: conjugate gradients to RELATIVE of
the load, started from the slots' mean widths and helped by the exact answer for the
grid whose every layer has its slots' mean stiffness (`_layer_solver`); the same
widths always give the same corners. When it has not settled in MOST_TURNS turns (sizes
differing by thousands of times, as in a chain of tears) the coordinate is solved
directly instead (slow: seconds at 32 places a side), so a layout always comes back;
in such a run cells do fold over each other in places (measured: 16 places, 825
tears, 513 collapses: up to 386 of 4,222 cells folded).
A width of 0 (a slot whose cell is gone) is held to FLOOR of the median width. The
universe wraps round; the drawing opens it at lattice index 0.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

RELATIVE = 1e-8
FLOOR = 1e-3
SHEAR = 0.3
MOST_TURNS = 300


def cell_layout(widths: np.ndarray) -> np.ndarray:
    """Corner points of the grid whose slots are as near these widths as one gapless grid allows."""
    widths = np.asarray(widths, dtype=np.float64)
    side = widths.shape[1]
    across = side + 1
    floor = FLOOR * float(np.median(widths[widths > 0.0])) if (widths > 0.0).any() else 1.0
    number = np.arange(across ** 3).reshape(across, across, across)
    rows, cols, stiff, loads, layers = [], [], [], [], []
    for axis in range(3):
        pad = [(0, 0) if other == axis else (1, 1) for other in range(3)]
        give = np.pad(1.0 / np.maximum(widths[axis], floor), pad)
        ones = np.pad(np.ones((side, side, side)), pad)
        weight, count = np.zeros_like(give), np.zeros_like(ones)
        first, second = [other for other in range(3) if other != axis]
        for shift_one in (0, 1):
            for shift_two in (0, 1):
                weight += np.roll(give, (shift_one, shift_two), axis=(first, second))
                count += np.roll(ones, (shift_one, shift_two), axis=(first, second))
        trim = tuple(slice(None) if other == axis else slice(1, None) for other in range(3))
        low = np.take(number, range(side), axis=axis).reshape(-1)
        high = np.take(number, range(1, across), axis=axis).reshape(-1)
        rows.append(low)
        cols.append(high)
        stiff.append(weight[trim].reshape(-1))
        loads.append(count[trim].reshape(-1))
        layers.append((weight[trim] / count[trim]).mean(axis=(first, second)))
    low, high = np.concatenate(rows), np.concatenate(cols)
    total = across ** 3
    guess = _mean_grid(widths)
    corners = np.zeros((across, across, across, 3))
    for axis in range(3):
        soft = [1.0 if other == axis else SHEAR for other in range(3)]
        spring = np.concatenate([soft[other] * stiff[other] for other in range(3)])
        matrix = coo_matrix((np.concatenate([spring, spring, -spring, -spring]),
                             (np.concatenate([low, high, low, high]), np.concatenate([low, high, high, low]))),
                            shape=(total, total)).tocsr()
        helper = _layer_solver([soft[other] * layers[other] for other in range(3)], across)
        load = np.zeros(total)
        np.add.at(load, rows[axis], -loads[axis])
        np.add.at(load, cols[axis], loads[axis])
        found = _settle(matrix, load, guess[..., axis].reshape(-1).copy(), helper)
        if found is None:
            found = _direct(matrix, load)
        corners[..., axis] = (found - found.mean()).reshape(across, across, across)
    return corners


def _layer_solver(layers: list, across: int):
    """The exact solve of the grid whose springs along each axis depend only on the layer (its slots' mean).

    That grid's matrix is a sum of three products of one-axis matrices, so it is solved by
    turning each axis to its own modes; the real grid differs from it only by the
    unevenness, which is why it is the helper (preconditioner) of the real solve.
    """
    share = np.full(across, 2.0)
    share[[0, -1]] = 1.0
    modes, tones = [], []
    for springs in layers:
        chain = np.zeros((across, across))
        at = np.arange(across - 1)
        chain[at, at] += springs
        chain[at + 1, at + 1] += springs
        chain[at, at + 1] -= springs
        chain[at + 1, at] -= springs
        tone, mode = np.linalg.eigh(chain / np.sqrt(np.outer(share, share)))
        modes.append(mode / np.sqrt(share)[:, None])
        tones.append(tone)
    total = tones[0][:, None, None] + tones[1][None, :, None] + tones[2][None, None, :]
    give = np.where(total > 1e-12 * total.max(), 1.0 / np.maximum(total, 1e-300), 0.0)

    def solve(load: np.ndarray) -> np.ndarray:
        turned = _turn(load.reshape(across, across, across), [mode.T for mode in modes]) * give
        return _turn(turned, modes).reshape(-1)
    return solve


def _settle(matrix, load: np.ndarray, found: np.ndarray, helper) -> np.ndarray | None:
    """Conjugate gradients with a helper: the corners' one coordinate that balances the load, or None when not settled.

    Sums are taken with einsum, not the threaded library: on a busy machine its threads
    wait on each other and a solve of 0.03 s takes 1.5 s (measured, 24 cores under load 64).
    """
    left = load - matrix @ found
    limit = RELATIVE * np.sqrt(np.einsum("i,i->", load, load))
    eased = helper(left)
    way = eased.copy()
    power = np.einsum("i,i->", left, eased)
    for _ in range(MOST_TURNS):
        if np.sqrt(np.einsum("i,i->", left, left)) <= limit:
            return found
        pushed = matrix @ way
        stride = power / np.einsum("i,i->", way, pushed)
        found += stride * way
        left -= stride * pushed
        eased = helper(left)
        power, before = np.einsum("i,i->", left, eased), power
        way = eased + (power / before) * way
    return None


def _direct(matrix, load: np.ndarray) -> np.ndarray:
    """The same balance by direct elimination, corner 0 held still (the grid may sit anywhere)."""
    hold = coo_matrix(([matrix.diagonal().max()], ([0], [0])), shape=matrix.shape)
    return spsolve((matrix + hold).tocsc(), load)


def _turn(field: np.ndarray, tables: list) -> np.ndarray:
    """Apply one square table along each of the three axes of a field."""
    field = np.einsum("ai,ijk->ajk", tables[0], field)
    field = np.einsum("bj,ajk->abk", tables[1], field)
    return np.einsum("ck,abk->abc", tables[2], field)


def _mean_grid(widths: np.ndarray) -> np.ndarray:
    """The straight grid whose every layer is as wide as its slots' mean: where the fit starts from."""
    side = widths.shape[1]
    corners = np.zeros((side + 1, side + 1, side + 1, 3))
    for axis in range(3):
        mean = widths[axis].mean(axis=tuple(other for other in range(3) if other != axis))
        edges = np.concatenate([[0.0], np.cumsum(mean)]) - 0.5 * mean.sum()
        shape = [side + 1 if other == axis else 1 for other in range(3)]
        corners[..., axis] = edges.reshape(shape)
    return corners
