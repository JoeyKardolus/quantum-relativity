"""A whole-step whirl and a flowing wave are never coded: their blocks stay FULL, bit-identical, windings kept.

A whirl line along z through the middle of a 64 x 64 x 16 grid of uniform energy
(phase = angle round the line, so flow 1/r): the blocks whose squares carry a whole turn,
and every block nearer than about 20 boxes (flow above 0.05), stay FULL and read back
exactly, the far blocks leave FULL (REUSED: the field repeats along z), and the phase winding of
every unit square over the whole grid is the same after switching. A plane wave of 0.3
rad per box (flow above the 0.05 tolerance) keeps every block FULL. A COARSE block given
flow by the core comes back FULL.
"""
from __future__ import annotations

import numpy as np

from physics.ladder import Level, LadderSettings, gather_blocks, load_region, measure_quietness, new_store
from physics.ladder import read_region, switch_levels, total_quanta, write_blocks
from physics.ladder.tests._fields import DEVICE
from physics.measure import phase_winding


def _complex(quanta: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """A complex field with |value|^2 = |quanta| and the box's phase, for the winding measure."""
    return np.sqrt(np.abs(quanta)) * np.exp(1j * phase.astype(np.float64))


def test_blocks_holding_a_whirl_or_flow_stay_full_and_bit_identical():
    gx, gy, gz = np.meshgrid(np.arange(64), np.arange(64), np.arange(16), indexing="ij")
    quanta = np.full(gx.shape, 2000, dtype=np.int32)
    phase = np.arctan2(gy - 31.5, gx - 31.5).astype(np.float32)
    store = new_store((8, 8, 2), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    coords = np.argwhere(store.level != Level.EMPTY)
    quiet = measure_quietness(store, coords)
    whirl_blocks = {tuple(block) for block, hits in zip(coords.tolist(), quiet["whirls"]) if hits > 0}
    assert {(bx, by) for bx, by, _ in whirl_blocks} == {(3, 3), (3, 4), (4, 3), (4, 4)}
    total = total_quanta(store)
    report = switch_levels(store)
    assert all(store.level[block] == Level.FULL for block in whirl_blocks)
    assert report["reused"] > 0 and store.level[0, 0, 0] == Level.REUSED
    read_q, read_p = read_region(store)
    near = np.hypot(gx - 31.5, gy - 31.5) < 16
    assert np.array_equal(read_q[near], quanta[near]) and np.array_equal(read_p[near], phase[near])
    assert np.array_equal(phase_winding(_complex(read_q, read_p)), phase_winding(_complex(quanta, phase)))
    assert total_quanta(store) == total

    wave = new_store((8, 8, 2), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(wave, quanta, (0.3 * gx).astype(np.float32))
    assert switch_levels(wave)["full"] == 8 * 8 * 2

    calm = new_store((8, 8, 2), LadderSettings(quiet_steps=1), device=DEVICE)
    grain = quanta + np.random.default_rng(3).integers(0, 4, size=gx.shape).astype(np.int32)
    load_region(calm, grain, np.zeros(gx.shape, np.float32))
    assert switch_levels(calm)["coarse"] == 8 * 8 * 2
    block = np.array([[1, 1, 0]])
    old_q, _ = gather_blocks(calm, block)
    slope = np.broadcast_to(0.3 * np.arange(8, dtype=np.float32)[:, None, None], (1, 8, 8, 8)).copy()
    assert write_blocks(calm, block, old_q.numpy(), slope) == 1
    switch_levels(calm)
    assert calm.level[1, 1, 0] == Level.FULL
    assert total_quanta(calm) == (int(grain.sum()), 0)
