"""Check step: axiom 2 read off the fluid itself, on the state as it stands after one step.

Contract: `check_step(state, settings) -> dict`; it writes nothing. A cell's going round
is read off its own boxes: between two neighbouring boxes of a ring (ring_box.py) fluid
flows from the wave that is ahead, `sqrt(q_a q_b) sin(phase_a - phase_b)` a unit of
local time, and a cell's `round_flow` is that summed over all three rings of all six
layers across its `swirl_axis`, counted in the cell's own `swirl_sense`. Nothing here
reads a label a rule wrote for itself except the two pull ledgers.
  against_volumes, against_joints   living cells holding energy whose own fluid does not
                      go round the way their sense says (`round_flow <= 0`); must be 0:
                      such a cell is no swirl
  unwound_volumes, unwound_joints   living cells holding energy in which the phase no
                      longer goes exactly once round the axis, the cell's way, in at
                      least four of its six layers (read on the middle ring)
  round_volumes, round_joints       the mean over those cells of `round_flow` per
                      quantum on the rings: how fast the fluid goes round (0.0 for none)
  joint_pull_made     living joints for which `spread_pull` put a source into the pull
                      this step (`pull_made`, the rule's own ledger); must be 0
  joint_pull_felt     living joints whose fluid the pull turned this step (`pull_felt`);
                      must be 0 (plus only pulls, and only plus is pulled)
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_2_swirls.tests.ring_box import RINGS, ring_box_host, ring_size_host
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, ENERGY_VOLUME, JOINT

_MIDDLE_RING = 1
_LAYERS_WOUND = 4


def _ring_places(ring: int) -> tuple:
    """The two coordinates across the axis of every box of a ring, in the order a swirl of sense +1 goes."""
    boxes = [ring_box_host(0, 0, ring, index) for index in range(ring_size_host(ring))]
    return np.array([box[1] for box in boxes]), np.array([box[2] for box in boxes])


def _across_axis(blocks: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Every cell's block turned so that its own swirl axis comes first: [cell, layer, first, second]."""
    turned = np.empty_like(blocks)
    for one in range(3):
        turned[axis == one] = np.moveaxis(blocks[axis == one], one + 1, 1)
    return turned


def _round_flow(amount: np.ndarray, phase: np.ndarray) -> tuple:
    """Per cell: the fluid's own flow round the axis in the sense +1, and the quanta on the rings' pairs."""
    flow = np.zeros(len(amount))
    weight = np.zeros(len(amount))
    for ring in range(RINGS):
        first, second = _ring_places(ring)
        root = np.sqrt(amount[:, :, first, second])
        angle = phase[:, :, first, second]
        pair = root * np.roll(root, -1, axis=2)
        flow += (pair * np.sin(angle - np.roll(angle, -1, axis=2))).sum(axis=(1, 2))
        weight += pair.sum(axis=(1, 2))
    return flow, weight


def _turns(phase: np.ndarray) -> np.ndarray:
    """Per cell and layer: the whole turns the phase makes going once round the middle ring in the sense +1."""
    first, second = _ring_places(_MIDDLE_RING)
    angle = phase[:, :, first, second]
    step = np.roll(angle, -1, axis=2) - angle
    step = (step + np.pi) % (2.0 * np.pi) - np.pi
    return np.rint(step.sum(axis=2) / (2.0 * np.pi)).astype(np.int64)


def _mean(values: np.ndarray) -> float:
    """The mean of some numbers; 0.0 for none."""
    return float(values.mean()) if values.size else 0.0


def check_step(state, settings) -> dict:
    """Count the cells whose own fluid does not go round their way, and any pull on or from minus."""
    count = state.count
    kind = state.host("kind")
    sense, axis = state.host("swirl_sense"), state.host("swirl_axis")
    holds = state.host("cell_quanta") > 0
    amount = _across_axis(np.abs(state.quanta.numpy()[:count]).astype(np.float64), axis)
    phase = _across_axis(state.phase.numpy()[:count], axis)
    flow, weight = _round_flow(amount, phase)
    flow = flow * sense
    wound = (_turns(phase) == -sense[:, None]).sum(axis=1) >= _LAYERS_WOUND
    counts = {}
    for name, cells in (("volumes", (kind == ENERGY_VOLUME) & holds), ("joints", (kind == JOINT) & holds)):
        counts["against_" + name] = int((cells & (flow <= 0.0)).sum())
        counts["unwound_" + name] = int((cells & ~wound).sum())
        counts["round_" + name] = _mean(flow[cells & (weight > 0)] / weight[cells & (weight > 0)])
    counts["joint_pull_made"] = int(((kind == JOINT) & (state.host("pull_made") != 0.0)).sum())
    counts["joint_pull_felt"] = int(((kind == JOINT) & (state.host("pull_felt") != 0.0)).sum())
    return counts
