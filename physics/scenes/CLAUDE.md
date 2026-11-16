# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each
scene builds a start, runs the cell engine (`physics.potts`) and returns measured numbers. Module map in
`__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler's 4 pi^2 / GM, the Hubble 73.5 / 67.4). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better; report it.
- **Only the engine's public face**: `new_universe`, `step`, `stats`, `Settings`,
  `cell_sums`, `contact_pairs` and `Universe.from_host` (with `phase`) for placed blobs and
  whirls; read only the Universe fields its contract lists (owner, psi through
  `wave_field.py`, kind, alive, labels, cell sums, cell_time, rest_a). Energy per box is |psi|^2 and
  flow is the phase's gradient (blob_start sets velocity as phase). Measures come from
  `physics.measure`, validated there. Kernels must not be reached into. Belt trick alone writes psi (its drive).
- **Local clocks.** Timescape (unfitted) and the fit time expansion by the densest cells'
  `cell_time` (`clocked_run.py`).
- **All rules on.** Scenes take `settings` (default `Settings.all_on()`); a rule switched off
  is only for a side-by-side run.
- **Box units.** Planck boxes are fixed; cells claim or give up boxes, so the occupied box
  count is the size of space and an energy radius says where the energy is.
- **Placed starts** (orbit, black hole, isotropy, spin, belt trick) break "nothing placed by hand" on
  purpose: each tests one mechanism. A blob is extra energy in an even start, never a ball in empty
  space. A blob must cover energy-volume boxes (all-odd cell places). The other scenes start from
  `new_universe`'s uneven field only.
- **Small and quick**: uneven starts default to 7^3 cells of 10 (bible: volumes start ~10
  across); placed starts keep 40 to 48 boxes a side. Tests run every scene on
  4 or 6 cells of 4 boxes and check only that the numbers are finite.

## Measured (2026-11-15, defaults, all rules on, cell engine)

Expansion (8^3 cells of 6^3, 60 steps): occupied x1.55, energy radius 23.7 to 22.5. Hubble gap: late / early 0.0007 (known 1.09): growth
stops at the grid's edge. Structure: contrast 1.97 to 2.8. Bound lump: no lump
forms. Orbit: the small blob falls in at distance 10 (early period 62, Newton 62.5) and turns
0.6 of a turn at 14 (early period 34, Newton 104). Black hole: no ball seals (core energy
about 100 to 155 of 250 to 2000; one collapse at 2000).
Isotropy: pull differs 2.0% by direction. Flow (speed 0.3 inside radius 16,
known shift 2.35 boxes), 4 seeds: carried 0.29 / 0.31 / 0.29 axis / face / body, sd up to
0.03, rest scatter 0.08: direction-free. Spin: winding lost, strength 0.86 to -0.12, tilt 90.
Belt trick (2026-11-16; own settings: 7^3 cells of 10, host on the pull's centre, 2 seeds):
the whirl dies before one turn and the web loses its start within 20 steps: no spin answer.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
PYTHONPATH=. .venv/bin/python -c "from physics.scenes import orbit; print(orbit()['kepler'])"
```
