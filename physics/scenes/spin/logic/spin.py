"""Spin scene: does a cell stay the one whole swirl it was born as while every rule runs?

Contract: `spin(cells_per_side=16, steps=2000, count=2, seed=1, settings=None,
start=None, device="cuda")` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a candidate's start amounts in). Nothing is placed:
every cell is born one whole swirl (axiom 2). It watches `count` energy volumes and
`count` joints (spin_cells), runs `steps` steps with every rule on and reads, at the
start and after every step, the whole turns of phase in each layer of the cell's own
block across its swirl axis (physics.measure cell_turns). A swirl lives inside its
cell, so only that block is read. Returns a dict (no verdict):
  cells_per_side, steps, seed, settings (dict)
  cases    one dict per watched cell: joint (1 a joint, 0 an energy volume), cell (id),
           axis (its swirl axis),
           expected    the turns it reads in every layer while it is its swirl
                       (physics.measure expected_turns of its axis and sense: +1 or -1)
           born        the turns read per layer at the start
           seen        every value read in any layer at any step, sorted
           held_steps  steps at which every layer read exactly `expected`
           first_off   the first step at which one did not (-1: never)
           end         the turns per layer at the end
Raises ValueError for a count the lattice has no room for.
"""
from __future__ import annotations

from dataclasses import asdict

from physics.engine import Grid, Settings, new_universe, step
from physics.measure import cell_turns, expected_turns
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.spin.definitions.spin_cells import spin_cells
from physics.scenes.spin.definitions.turn_record import turn_record


def _layer_turns(universe: Grid, cell: int, axis: int) -> list:
    """The whole turns in each layer of one cell's block across `axis`, now."""
    return [int(turns) for turns in cell_turns(universe.block("quanta", cell), universe.block("phase", cell), axis)]


def spin(cells_per_side: int = 16, steps: int = 2000, count: int = 2, seed: int = 1,
         settings: Settings | None = None, start: dict | None = None, device: str = "cuda") -> dict:
    """Per watched cell: the turns read across its swirl axis at every step of a run with all rules on."""
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    volumes, joints = spin_cells(universe.side, count)
    axis, sense = universe.host("swirl_axis"), universe.host("swirl_sense")
    cases = [{"joint": int(cell in joints), "cell": cell, "axis": int(axis[cell]),
              "expected": expected_turns(int(axis[cell]), int(sense[cell]))} for cell in volumes + joints]
    for case in cases:
        case["born"] = _layer_turns(universe, case["cell"], case["axis"])
    readings = [[] for _ in cases]
    for _ in range(steps):
        step(universe, settings)
        for case, seen in zip(cases, readings):
            seen.append(_layer_turns(universe, case["cell"], case["axis"]))
    for case, seen in zip(cases, readings):
        case.update(turn_record(seen, case["expected"]))
        case["end"] = seen[-1] if seen else case["born"]
    return {"cells_per_side": cells_per_side, "steps": steps, "seed": seed, "settings": asdict(settings), "cases": cases}
