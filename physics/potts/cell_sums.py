"""Cell sums: each cell's energy, size, potential, wanted size, compression and clock.

Contract: `cell_sums(universe, settings)` rewrites, on the GPU, for every living cell:
  cell_energy     its boxes' energy (int64 quanta, exact)
  cell_boxes      its box count: its size (bible, "How the engine holds the foam")
  cell_potential  its boxes' summed pull potential; cell_potential_mean = sum / boxes
  cell_want       the box count its energy wants (rule 4, more energy, smaller):
                  labels * base_boxes * (labels * rest / E)**size_power, rest the
                  start's mean cell energy of its kind (cell_rest_a, cell_rest_b)
  cell_compression  want / boxes: above 1 the cell is pressed smaller than its energy
                  wants (a squeezed joint)
  cell_clock      (s + 1) / (s + q), s = clock_scale, q its energy per box over its
                  kind's start mean per box (rest_a, rest_b): the more packed, the
                  slower (bible: time is local, joints included)
and a_quanta, the A-energy per box (the pull's source). A dead or empty cell keeps
zeros and clock 1. Integer sums are exact; the potential sum is float atomics, so its
last bits depend on the order threads ran. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)


@wp.kernel
def _clear(cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
           cell_potential: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    cell_energy[cell] = wp.int64(0)
    cell_boxes[cell] = 0
    cell_potential[cell] = 0.0


@wp.kernel
def _gather(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
            potential: wp.array3d(dtype=wp.float32), kind: wp.array(dtype=wp.int32),
            cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
            cell_potential: wp.array(dtype=wp.float32), a_quanta: wp.array3d(dtype=wp.int64)):
    i, j, k = wp.tid()
    cell = owner[i, j, k]
    a_quanta[i, j, k] = wp.int64(0)
    if cell < 0:
        return
    wp.atomic_add(cell_energy, cell, energy[i, j, k])
    wp.atomic_add(cell_boxes, cell, 1)
    wp.atomic_add(cell_potential, cell, potential[i, j, k])
    if kind[cell] == 0:
        a_quanta[i, j, k] = energy[i, j, k]


@wp.kernel
def _derive(kind: wp.array(dtype=wp.int32), labels: wp.array(dtype=wp.int32),
            cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
            cell_potential: wp.array(dtype=wp.float32), cell_rest_a: float, cell_rest_b: float,
            rest_a: float, rest_b: float, base_boxes: float, size_power: float, clock_scale: float,
            cell_want: wp.array(dtype=wp.float32), cell_compression: wp.array(dtype=wp.float32),
            cell_clock: wp.array(dtype=wp.float32), cell_potential_mean: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    boxes = float(cell_boxes[cell])
    if boxes <= 0.0:
        cell_want[cell] = 0.0
        cell_compression[cell] = 0.0
        cell_clock[cell] = 1.0
        cell_potential_mean[cell] = 0.0
        return
    held = wp.max(float(cell_energy[cell]) * _QUANTUM, _QUANTUM)
    cell_rest = cell_rest_a
    box_rest = rest_a
    if kind[cell] == 1:
        cell_rest = cell_rest_b
        box_rest = rest_b
    count = float(labels[cell])
    want = count * base_boxes * wp.pow(count * cell_rest / held, size_power)
    cell_want[cell] = want
    cell_compression[cell] = want / boxes
    concentration = held / (boxes * box_rest)
    cell_clock[cell] = (clock_scale + 1.0) / (clock_scale + concentration)
    cell_potential_mean[cell] = cell_potential[cell] / boxes


def cell_sums(universe: Universe, settings: Settings) -> None:
    """Sum every cell's energy, boxes and potential; derive its want, compression and clock."""
    device = universe.owner.device
    wp.launch(_clear, dim=universe.cell_count, device=device,
              inputs=[universe.cell_energy, universe.cell_boxes, universe.cell_potential])
    wp.launch(_gather, dim=(universe.side,) * 3, device=device,
              inputs=[universe.owner, universe.energy, universe.potential, universe.kind, universe.cell_energy,
                      universe.cell_boxes, universe.cell_potential, universe.a_quanta])
    wp.launch(_derive, dim=universe.cell_count, device=device,
              inputs=[universe.kind, universe.labels, universe.cell_energy, universe.cell_boxes,
                      universe.cell_potential, universe.cell_rest_a, universe.cell_rest_b, universe.rest_a,
                      universe.rest_b, float(universe.base_boxes), settings.size_power, settings.clock_scale,
                      universe.cell_want, universe.cell_compression, universe.cell_clock,
                      universe.cell_potential_mean])
