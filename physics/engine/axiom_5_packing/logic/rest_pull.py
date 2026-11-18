"""Rest pull: the pull at rest for the energy as it lies, found from the pull as it stands.

Contract: `rest_pull(state, settings)` writes `pull` and `pull_old` (equal: the field
stands still) as the pull at rest for the energy as it lies: the field one wave step
(`wave_step`, pull_wave_law.py) moves by the same amount in every cell, with mean 0 over
the ids in use. So for every cell that holds energy and hears a neighbour
  sum over its neighbours of distance * (neighbour's pull - own pull)
      = neighbours * (4 pi G quanta / 6 + level * reach / weight)
with the distances, `reach`, `wave_weight` and `pull_source` as the wave step uses them
(the same numbers, worked on the host in float64) and `level` the one number that lets
the sum over all cells close: the even pull-back a wrapping universe needs, spread by
the room each cell stands in. It is solved on the host by conjugate gradients, each
cell's unknown scaled by its neighbours' summed distance, starting from the pull as it
stands (so a caller that changed the energy a little is nearly there already), until no
cell is further from rest than a ten-millionth of the pull's own spread, at most `40 * side
+ 400` rounds. `state.pull_unrest` keeps how far the furthest cell still is, in units of
the pull. A cell that holds nothing, or hears nobody, keeps the pull it had. Needs
`length` (so it runs after `size_cells`). Changes no quantum and no clock.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

from physics.engine.axiom_2_swirls.definitions.sign_law import makes_pull_host
from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL

_ROUNDS_PER_SIDE = 40
_ROUNDS_MORE = 400
_STILL = 1e-7


def _heard_pairs(state, alive: np.ndarray) -> tuple:
    """Every neighbour entry between two cells that hold energy: the hearing cell and the one it hears."""
    count = int(state.count)
    number = state.adj_count[:count].astype(np.int64)
    hearer = np.repeat(np.arange(count, dtype=np.int64), number)
    first = np.repeat(state.adj_start[:count].astype(np.int64) - (np.cumsum(number) - number), number)
    heard = state.pool_id[first + np.arange(hearer.size, dtype=np.int64)].astype(np.int64)
    both = alive[hearer] & alive[heard]
    return hearer[both], heard[both]


def _distances(length: np.ndarray, hearer: np.ndarray, heard: np.ndarray) -> np.ndarray:
    """Half the one length plus half the other, for every pair."""
    return (length[hearer] + length[heard]) / 2.0


def _solve(hearer: np.ndarray, heard: np.ndarray, weight: np.ndarray, wanted: np.ndarray, pull: np.ndarray,
           rounds: int) -> float:
    """Conjugate gradients on `sum of weight * (own - neighbour) = wanted`, in place; returns the last unrest."""
    cells = pull.size
    pairs = csr_matrix((weight, (hearer, heard)), shape=(cells, cells))
    summed = np.asarray(pairs.sum(axis=1)).reshape(-1)
    hears = summed > 0.0
    scale = np.where(hears, summed, 1.0)

    def apply(field: np.ndarray) -> np.ndarray:
        return summed * field - pairs @ field

    left = np.where(hears, wanted - apply(pull), 0.0)
    scaled = left / scale
    way = scaled.copy()
    along = float(left @ scaled)
    unrest = float(np.abs(scaled).max(initial=0.0))
    for _ in range(rounds):
        spread = float(pull[hears].max(initial=0.0) - pull[hears].min(initial=0.0))
        if unrest <= _STILL * spread or along == 0.0:
            break
        pushed = apply(way)
        stride = along / float(way @ pushed)
        pull += stride * way
        left -= stride * pushed
        scaled = left / scale
        after = float(left @ scaled)
        way = scaled + (after / along) * way
        along = after
        unrest = float(np.abs(scaled).max(initial=0.0))
    return unrest


def rest_pull(state, settings) -> None:
    """Solve the pull, from where it stands, to the field at rest for the energy as it lies."""
    count = int(state.count)
    kind, quanta = state.host("kind"), state.host("cell_quanta")
    alive = (kind != NO_CELL) & (quanta > 0) & np.isfinite(state.host("length"))
    hearer, heard = _heard_pairs(state, alive)
    distance = _distances(state.host("length").astype(np.float64), hearer, heard)
    hears = np.bincount(hearer, minlength=count).astype(np.float64)
    reach = np.bincount(hearer, weights=distance, minlength=count) / np.maximum(hears, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(reach > 0.0, np.minimum((settings.light_speed * settings.step_time / reach) ** 2, 1.0), 1.0)
    kinds = np.unique(kind)
    plus = np.isin(kind, kinds[[makes_pull_host(int(one)) for one in kinds]])
    made = np.where(plus & alive, 4.0 * math.pi * settings.pull_strength * quanta.astype(np.float64) / 6.0, 0.0)
    room = reach / weight
    level = -float((hears * made).sum() / (hears * room).sum()) if hears.sum() else 0.0
    whole = state.pull.numpy()
    pull = whole[:count].astype(np.float64)
    state.pull_unrest = _solve(hearer, heard, distance, -hears * (made + level * room), pull,
                               _ROUNDS_PER_SIDE * state.side + _ROUNDS_MORE)
    pull -= pull.mean()
    whole[:count] = pull
    state.pull.assign(whole)
    state.pull_old.assign(whole)
