# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each
scene builds a start, runs the engine (`physics.engine`) and returns measured numbers. Module map in
`__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler's 4 pi^2 / GM, the Hubble 73.5 / 67.4). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better; report it.
- **Only the engine's public face**: `new_universe`, `step`, `stats`, `Settings`, `cells`,
  `host_boxes`, `State.from_host` (signed quanta and phase) for placed blobs and whirls, and
  `state.start`. Energy per box is |quanta| (whole numbers, plus pulls, minus pushes), the
  fluid sqrt(|quanta|) e^(i phase) (`wave_field.py`), flow the phase's gradient. Energies
  passed to blob_start are in units of 1000 quanta (the start's quanta per box). Cells are
  read off the signs each sample (`engine/cells.py`: one joint per neighbour pair); ids do
  not persist. Kernels must not be reached into. Belt trick alone writes phase (its drive).
- **Local clocks.** Timescape (unfitted) and the fit time expansion by the most packed
  energy volumes' mean box clock, volumes followed by shared boxes (`clocked_run.py`).
- **All rules on.** Scenes take `settings` (default `Settings()`); a rule switched off
  is only for a side-by-side run.
- **Box units.** Planck boxes are fixed; claims move boxes between kinds, so the occupied box
  count is the size of space and an energy radius says where the energy is.
- **Placed starts** (orbit, black hole, isotropy, spin, belt trick) break "nothing placed by hand" on
  purpose: each tests one mechanism. A blob is extra energy in an even start, never a ball in empty
  space. A blob must cover energy-volume boxes (all-odd cell places). The other scenes start from
  `new_universe`'s uneven field only.
- **Small and quick**: uneven starts default to 7^3 cells of 10 (bible: volumes start ~10
  across); placed starts keep 40 to 48 boxes a side. Tests run every scene on
  4 or 6 cells of 4 boxes and check only that the numbers are finite.

## Measured (2026-11-16, defaults, seed 1, whole-quanta engine; step 0.25)

Expansion (8^3 of 6^3, 60 steps): occupied x1.22, 11,437 boxes collapsed, drift 0. Hubble
late/early 3.2 (known 1.09). No lump forms. Orbit: collapse breaks the excess tracker, no
orbit. Black hole: nothing seals. Pull axis vs diagonal 0.2-2% to r = 10, more outward
(pull 0 outside the cube). Flow carried 0.898 / 0.898 / 0.899. Spin: winding exact 50 of
60 steps. Timescape: one observer left at 7^3; at 20^3 cells (216M boxes, 729
volumes, 8 min, host measures ~10 s a sample) three observers, clocks 0.88-0.90 of the
counter, q -82 by the counter and -84 by theirs. Belt trick: the whirl keeps winding 1 through both turns;
the web's twist after one and two turns (0.06, -0.10) is the still run's (-0.01, -0.10).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
PYTHONPATH=. .venv/bin/python -c "from physics.scenes import orbit; print(orbit()['kepler'])"
```
