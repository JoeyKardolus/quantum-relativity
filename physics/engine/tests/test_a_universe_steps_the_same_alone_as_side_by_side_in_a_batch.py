"""Universes side by side in one GPU job stay apart: each keeps its amounts exact and steps as it would alone.

Issue #3 (many small universes side by side in one GPU job; the harness physics/bench).
A batch of 4 universes from new_universe (4 cells a side of 6 boxes, seeds 5..8, every
rule on) runs 120 steps in one State. Each universe's plus-and-minus amount is identical
to its start at the end, and universe 2 (seed 7) ends bit for bit as the same universe
stepped alone with seed 7: no rule reads across from one universe to the next, and the
whole-number rounding is seeded per universe. The same seed stepped twice ends bit for
bit the same (the engine is repeatable).
"""
import numpy as np

from physics.engine import Settings, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_a_batched_universe_matches_its_twin_and_keeps_its_amount():
    settings = Settings()
    batch = new_universe(4, 6, seed=5, universes=4)
    alone = new_universe(4, 6, seed=7)
    again = new_universe(4, 6, seed=7)
    start = [row["amount"] for row in totals(batch)["universes"]]
    for _ in range(120):
        step(batch, settings)
        step(alone, settings)
        step(again, settings)
    end = [row["amount"] for row in totals(batch)["universes"]]
    print(f"amounts at the start {start}, at the end {end}; collapsed per universe {batch.tallies.numpy()[:, 1].tolist()}")
    assert end == start
    twin, single, repeat = host_boxes(batch, 2), host_boxes(alone), host_boxes(again)
    assert np.array_equal(twin["quanta"], single["quanta"]) and np.array_equal(twin["phase"], single["phase"])
    assert np.array_equal(single["quanta"], repeat["quanta"]) and np.array_equal(single["phase"], repeat["phase"])
