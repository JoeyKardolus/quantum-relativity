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
- **Small and quick**: uneven starts default to 7^3 cells of 10 (bible: volumes start ~10
  across); placed starts keep 40 to 48 boxes a side. Tests run every scene on
  5 to 7 lattice places of 6 boxes (a cell needs `least_cell_boxes`, 216) and check only that the numbers are finite (a place
  count n gives (n - 1) // 2 volumes a side without wrapping).

## Measured

Every number from before the 2026-11-16 rebuild (cell graph) is void: re-measure before
quoting. Last pre-rebuild reading (8^3 of 6^3, 60 steps): occupied x1.22, growth rate
slowing, no lump, no orbit, nothing seals. Post-rebuild default run: issue #1.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
PYTHONPATH=. .venv/bin/python -c "from physics.scenes import orbit; print(orbit()['kepler'])"
```
