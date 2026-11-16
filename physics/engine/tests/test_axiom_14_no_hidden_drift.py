"""Axiom 14: no hidden drift: growth of space comes only from the rules, with no spread setting; growth is measured inside too.

Bible rule 6 ("there is no separate drift setting") and rule 8 (drift from imperfect
transfer is friction, not a share). Default universe (seed 1, Settings(), axiom_run.py),
300 steps:
  - Settings carries no spread setting: no field whose name holds "spread" or "drift"
    (the engine has no rule that spreads quanta by a share, so there is no spread
    tally to read either; before 2026-11-16 this also read that tally)
  - every quantum that crossed a joint left one volume and arrived in the other: over
    the run the plus total changed only by collapse and tears (axiom 02's sum, read
    here again) while the crossings moved quanta (tallies' crossed > 0)
  - growth inside the universe, measured and printed beside the edge's: the boxes held
    by the inner cells (lattice place 3 .. n - 2 on every axis) and by the others at
    step 300 over step 0 (the default universe wraps round, so no cell has an edge;
    the box total cannot change and the split is printed only)
"""
import dataclasses

import numpy as np

from physics.engine import Settings
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
