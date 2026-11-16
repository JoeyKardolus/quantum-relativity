"""Axiom 8, Nothing else: no rule beyond the eight acts: no hidden drift, and no universe reaches into another.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16), checked in 2 parts, each
kept whole from the file it came from (names that two parts shared carry a _P<n> suffix).

Part 1 (was axiom 14). No hidden drift: growth of space comes only from the rules, with no spread setting; growth is measured inside too.

Bible rule 6 ("there is no separate drift setting") and rule 8 (drift from imperfect
transfer is friction, not a share). Default universe (seed 1, Settings(), axiom_run.py),
300 steps:
  - Settings carries no spread setting: no field whose name holds "spread" or "drift"
    (the engine has no rule that spreads quanta by a share, so there is no spread
    tally to read either; before 2026-11-16 this also read that tally)
  - every quantum that crossed a joint left one volume and arrived in the other: over
    the run the plus total changed only by collapse and tears (axiom 6's sum, read
    here again) while the crossings moved quanta (tallies' crossed > 0)
  - growth inside the universe, measured and printed beside the edge's: the boxes held
    by the inner cells (lattice place 3 .. n - 2 on every axis) and by the others at
    step 300 over step 0 (the default universe wraps round, so no cell has an edge;
    the box total cannot change and the split is printed only)

Part 2 (was test_a_universe_steps_the_same_alone_as_side_by_side_in_a_batch). Universes side by side in one GPU job stay apart: each keeps its amounts exact and steps as it would alone.

Issue #3 (many small universes side by side in one GPU job; the harness physics/bench).
A batch of 4 universes from new_universe (4 cells a side of 6 boxes, seeds 5..8, every
rule on) runs 120 steps in one State. Each universe's plus-and-minus amount is identical
to its start at the end, and universe 2 (seed 7) ends bit for bit as the same universe
stepped alone with seed 7: no rule reads across from one universe to the next, and the
whole-number rounding is seeded per universe. The same seed stepped twice ends bit for
bit the same (the engine is repeatable).
"""
import dataclasses
import numpy as np

from physics.engine import Settings, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_space_grows_only_from_the_rules():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    names = [field.name for field in dataclasses.fields(Settings)]
    hidden = [name for name in names if "spread" in name or "drift" in name]
    crossed = run["steps"][-1]["tallies"][0]
    unexplained = abs((run["steps"][-1]["plus"] - run["start"]["plus"])
                      - (run["steps"][-1]["tallies"][3] - run["steps"][-1]["torn"]))
    first, last = run["checks"][0]["cells"], run["checks"][-1]["cells"]
    cells_per_axis = first["place"].max()
    inner = np.all((first["place"] >= 3) & (first["place"] <= cells_per_axis - 2), axis=1)
    inner_growth = last["boxes"][inner].sum() / first["boxes"][inner].sum()
    outer_growth = last["boxes"][~inner].sum() / first["boxes"][~inner].sum()
    print(f"spread or drift settings: {hidden}; quanta crossed over 300 steps: {crossed}; plus-total change "
          f"not explained by collapse or tears: {unexplained}; boxes at step 300 over step 0: inner cells "
          f"{inner_growth:.3f} ({int(first['boxes'][inner].sum())} -> {int(last['boxes'][inner].sum())}), outer layer "
          f"{outer_growth:.3f} ({int(first['boxes'][~inner].sum())} -> {int(last['boxes'][~inner].sum())})")
    assert not hidden
    assert crossed > 0 and unexplained == 0


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
