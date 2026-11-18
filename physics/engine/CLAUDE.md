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

Each holds `definitions/`, `logic/`, `tests/` and its own `CLAUDE.md` ("Assumed", "Not done", "Open").

## Must not get wrong

- **Nothing loose**: subdirectories plus only `__init__.py`, `conftest.py`, `CLAUDE.md`.
- **`__init__.py` here is the only file that calls several axioms' logic**: the public
  names, `ARRAYS`, `HOST`, `Settings`, `new_universe` and `step`.
- **Definitions never import logic**, and import only axioms earlier in the order
  3, 2, 1, 7, 5, 4, 6. Logic never imports another axiom's logic.
- **One state object**, `Grid` (axiom 3), built from every axiom's `ARRAYS` and `HOST`.
- **Never**: boxes as space, links as springs, a base push, a drift setting, a charge sign, a size at rest, an instant pull, a seal, a rate setting for crossing, an off switch in a setting, a rule no axiom asks for, a default tuned to look good.
- **One fluid, one phase field** (`pair_rotation`), inside a cell and across a joint alike.
- **The step's order** is in `step`: clocks, pull, fluid, crossing, splits, press, making, events, the one settling. `step(..., quantum=False)` is phase 1 of the fit: the layers below axiom 7 (no split opened, carried or settled).
- **The start is at rest in balance** (`lay_start`): the pull at rest, every joint holding the energy at which its push equals the pull across it (no setting says what a joint holds), every cell a standing swirl that mirrors its neighbours. A scene that places energy calls `lay_start` again.

## Tests

`tests/conftest.py` makes five real runs (`run_records`), 16 a side: `default` (stands), `collapsing` (a joint collapses a hundredth above fresh), `uneven` (tears), `held` (push 40) and `gentle` (`held`, pull a hundred times weaker). Every step runs each axiom's `tests/check_step.py`, the books and `check_pattern`. A failing test is a clause the engine does not keep; never bend one to pass. 16 are expected failures (see "Not done" in the axioms): swirls lost through events and in joints of one quantum a box (9), the fuller volumes gaining with the pull as good as off (2), no lump growing in 40 steps (1), the collapsing run running empty with short splits (4).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # needs CUDA
```
