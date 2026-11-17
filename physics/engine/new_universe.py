"""New universe: a perfect A B A B grid of cells, every energy volume wrapped in joints, slightly uneven in energy.

Contract: `new_universe(cells_per_side=32, seed=1, unevenness=0.1, slope=-3.0,
box_quanta=3000, joint_quanta=100, inner_unevenness=0.1, volume_length=10.0,
joint_length=1.0, spare=0.25, device="cuda")` returns a State (state.py) at step 0, phase
0, pull 0. The lattice has cells_per_side places per axis (even, at least 4) and wraps
round: a place is an energy volume when all three indices are even and a joint
everywhere else (faces, edges, corners), so every volume is wrapped by joints on all 26
sides and the joints are one web. Energy: the uneven field g over the lattice
(uneven_field.py: seed, slope, rms unevenness, ripples at every size; plain seeded
noise below 8 places a side) gives a volume box_quanta * exp(g) / mean(exp(g)) quanta
per box on average, a joint joint_quanta; each box then holds that times exp(inner
unevenness * a standard normal) (a hash of the box and the seed), rounded to whole
quanta, at least 1, minus in a joint, so the fluid inside every cell starts moving at
once. The default unevenness is slight (0.1): at 0.3 one volume in 70 starts longer
than the default tear length and the run opens with tears. The size law's two constants are fixed here, once: an energy volume of the
start's mean volume quanta is volume_length long and a joint of the start's mean joint
quanta joint_length (thin mortar holding a small share of the energy: a fifth at
the defaults; a joint needs about 100 quanta a box to hold a whirl, measured: at 30 a
placed whirl was lost by step 268 of 1,000, at 100 it held); from then on a
cell's length is constant / its quanta and nothing else (cell_laws.py, run here with
the default Settings so lengths and clock rates are set). `spare` is the share of extra
cell ids kept ready for tears (the state grows when they run out). Built on the GPU;
only the lattice-sized field is made on the host.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.cell_laws import cell_laws
from physics.engine.settings import Settings
from physics.engine.state import BLOCK, BOXES, ENERGY_VOLUME, JOINT, State
from physics.engine.totals import totals
from physics.engine.uneven_field import uneven_field


@wp.kernel
def _fill(kind: wp.array(dtype=wp.int32), base: wp.array(dtype=wp.float32), inner: float, seed: int, boxes: int,
          across: int, quanta: wp.array4d(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64)):
    cell, one, two, three = wp.tid()
    rng = wp.rand_init(seed, cell * boxes + (one * across + two) * across + three)
    amount = wp.max(int(base[cell] * wp.exp(inner * wp.randn(rng)) + 0.5), 1)
    quanta[cell, one, two, three] = kind[cell] * amount
    wp.atomic_add(cell_quanta, cell, wp.int64(amount))


def new_universe(cells_per_side: int = 32, seed: int = 1, unevenness: float = 0.1, slope: float = -3.0,
                 box_quanta: int = 3000, joint_quanta: int = 100, inner_unevenness: float = 0.1,
                 volume_length: float = 10.0, joint_length: float = 1.0, spare: float = 0.25,
                 device: str = "cuda") -> State:
    """Build the start: the perfect grid of energy volumes and joints with slightly uneven energy."""
    if box_quanta < 3 or joint_quanta < 3:
        raise ValueError("box_quanta and joint_quanta must be at least 3")
    side = int(cells_per_side)
    state = State(side, int(side ** 3 * (1.0 + spare)) + 2, seed=seed, device=device)
    if side >= 8:
        field = uneven_field(side, seed, slope, unevenness)
    else:
        field = np.random.default_rng(seed).standard_normal((side, side, side)) * unevenness
    volumes = state.kind_h[:state.lattice] == ENERGY_VOLUME
    weight = np.exp(field).reshape(-1)
    base = np.where(volumes, box_quanta * weight / weight[volumes].mean(), float(joint_quanta))
    base_d = wp.array(base.astype(np.float32), dtype=wp.float32, device=device)
    wp.launch(_fill, dim=(state.lattice, BLOCK, BLOCK, BLOCK), device=device,
              inputs=[state.kind, base_d, float(inner_unevenness), int(seed), BOXES, BLOCK, state.quanta,
                      state.cell_quanta])
    held = state.host("cell_quanta")
    for sort, long in ((ENERGY_VOLUME, volume_length), (JOINT, joint_length)):
        state.start_quanta[sort] = float(held[state.kind_h[:state.lattice] == sort].mean())
        state.start_length[sort] = float(long)
        state.size_energy[sort] = float(long) * state.start_quanta[sort]
    cell_laws(state, Settings())
    state.start = totals(state)
    state.start["size"] = float(state.host("length").astype(np.float64).sum())
    return state
