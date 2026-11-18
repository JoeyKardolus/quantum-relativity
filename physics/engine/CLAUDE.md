# physics/engine/

The engine, built on the owner's seven axioms (`docs/bible.md#physicsquantum-relativity`). Their words
win over this file and the code. One directory per axiom:

- `axiom_1_energy/`: one always-moving quantum fluid in whole signed quanta; exact books.
- `axiom_2_swirls/`: every cell's energy goes round in whole steps; plus pulls, minus pushes.
- `axiom_3_structure/`: the cells are the grid; relations and a size set by energy alone.
- `axiom_4_crossing/`: the fluid's own step taken across a joint, only through joints; friction is heat.
- `axiom_5_packing/`: more packed is smaller, slower, pulls harder; the pull is a wave.
- `axiom_6_limits/`: squeezed joints make energy and collapse; overstretched volumes tear.
- `axiom_7_quantum/`: phase everywhere; outflow that meets a fork splits, goes on, loses its momentum, settles whole.

Each holds `definitions/`, `logic/`, `tests/` and its own `CLAUDE.md`: "Assumed", "Not done"
and "Open" say what the axiom left open and what the engine does not do.

## Must not get wrong

- **Nothing loose.** A directory here holds subdirectories plus only `__init__.py`,
  `conftest.py` and `CLAUDE.md`. There is no axiom 8.
- **`__init__.py` here is the only file that calls several axioms' logic**: the public
  names, `ARRAYS`, `HOST`, `Settings`, `new_universe` and `step`.
- **Definitions never import logic**, and import only axioms earlier in the order
  3, 2, 1, 7, 5, 4, 6. Logic never imports another axiom's logic.
- **One state object**, `Grid` (axiom 3), built from every axiom's `ARRAYS` and `HOST`.
- **Never**: boxes as space (a cell's block is its own for good), links as springs, a base push, a drift setting, a charge
  sign, a size at rest, an instant pull, a seal, a rate setting for crossing, an off
  switch in a setting, a rule no axiom asks for, a default tuned to make a result look good.
- **One fluid, one phase field.** Fluid flows from the wave whose phase is ahead (`pair_rotation`),
  inside a cell and across a joint alike; `cross` reads two cells' phases as they stand. So a
  rule that turns phases (`pull_fluid`) must turn alike on both sides of a joint. No law invents
  fluid beyond a cell's side: that made motion out of nothing.
- **The step's order** is in `step`: clocks, pull, fluid, crossing, splits, press, making, events, the one settling.

## Tests

`tests/conftest.py` makes four real runs (`run_records`), 16 a side: `default` (collapses), `uneven` (tears), `held` (the grid stands) and `gentle` (`held`, the pull a hundred times weaker). Every step runs each axiom's `tests/check_step.py`, the books and `check_pattern`; each
axiom's `tests/` asserts on those. A failing test is a clause the engine does not keep; never bend one to pass. 20 are marked `NOT_KEPT` (strict expected failures): cells lose their swirl (10), the pull outruns light (6) and turns a box
over half a turn a step (2), and the fuller volumes gain with the pull as good as off (2);
see "Not done" in axioms 2 and 5.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # needs CUDA
```
