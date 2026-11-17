# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each
scene builds a start, runs the engine (`physics.engine`) and returns measured numbers. Module map in
`__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler's 4 pi^2 / GM, the Hubble 73.5 / 67.4). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better; report it.
- **Only the engine's public face**: `new_universe`, `step`, `stats`, `Settings`, `cells`,
  `host_boxes`, `State.from_host` (signed quanta, phase and `lattice=lattice_of(even)`) for
  placed blobs and whirls, and `state.start`. Energy per box is |quanta| (whole numbers, plus pulls, minus pushes), the
  fluid sqrt(|quanta|) e^(i phase) (`wave_field.py`), flow the phase's gradient. Energies
  passed to blob_start are in units of 1000 quanta (the start's quanta per box). Cells are
  the engine's cell graph (`engine/cells.py`: one joint per pair of neighbouring volumes);
  collapse and split change it. Kernels must not be reached into. Belt trick alone writes phase (its drive).
- **Local clocks.** Timescape (unfitted) times expansion by the most packed
  energy volumes' mean box clock, volumes followed by shared boxes (`clocked_run.py`); the fit by the clock where the typical quantum sits.
- **All rules on.** Scenes take `settings` (default `Settings()`); a rule switched off
  is only for a side-by-side run.
- **Open space.** Growth scenes and blob starts pass `margin=open_margin(...)` (the fit wraps and measures inside, `fit/fit_measure.py`);
  `new_universe`'s own margin 0 wraps or has walls, where nothing grows outward.
- **Box units.** Planck boxes are fixed; boxes change cell as cells follow their sizes, so the
  occupied box count is the size of space and an energy radius says where the energy is.
- **Placed starts** (orbit, black hole, isotropy, spin, belt trick) break "nothing placed by hand" on
  purpose: each tests one mechanism. A blob is extra energy in an even start, never a ball in empty
  space. A blob must cover energy-volume boxes (all-odd cell places). The other scenes start from
  `new_universe`'s uneven field only.
- **Orbit's blob is extra energy of whole volumes** (box jiggle outweighs it), lost at twice its start's.
- **Small and quick**: uneven starts default to 7^3 cells of 10; placed starts keep 40 to
  48 boxes a side. Tests run every scene on 5 to 7 lattice places of 6 boxes and check
  only that the numbers are finite (n places give (n - 1) // 2 volumes a side unwrapped).

## Measured (2026-11-17, seed 1, pass marks written first)

Defaults and the fitted file (void): 2 of 13 pass in each, dimension (2.84, as a plain
grid) and the staged quantum split. Misses:
orbit's blob lost after 15 and 96 steps; bound lump x1.25, the
rest x1.32; black hole 0 of 4 sealed; Hubble late/early 0.023; timescape q
21.2 by the counter, 20.6 by lump clocks; the whirl holds one turn 107 steps; the belt
trick's web turns by itself.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
```
