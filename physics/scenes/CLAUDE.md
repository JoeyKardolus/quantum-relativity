# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`, "How results are trusted"): each
scene builds a start, runs the cell engine (`physics.potts`) and returns measured numbers. Module map in
`__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler's 4 pi^2 / GM, the Hubble 73.5 / 67.4). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better; report it.
- **Only the engine's public face**: `new_universe`, `step`, `stats`, `Settings`,
  `cell_sums`, `contact_pairs` and `Universe.from_host` (with `phase`) for placed blobs and
  whirls; read only the Universe fields its contract lists (owner, psi through
  `wave_field.py`, kind, alive, labels, cell sums, rest_a). Energy per box is |psi|^2 and
  flow is the phase's gradient (blob_start sets velocity as phase). Measures come from
  `physics.measure`, validated there. Kernels must not be reached into.
- **All rules on.** Scenes take `settings` (default `Settings.all_on()`); a rule switched off
  is only for a side-by-side run.
- **Box units.** Planck boxes are fixed; cells claim or give up boxes, so the occupied box
  count is the size of space and an energy radius says where the energy is.
- **Placed starts** (orbit, black hole, isotropy, spin) break "nothing placed by hand" on
  purpose: each tests one mechanism. A blob is extra energy in an even start, never a ball in empty
  space (empty space has no boxes to flow through). A blob must cover energy-volume boxes, and
  those sit only at all-odd cell places (orbit's defaults are chosen for that). The other scenes start from
  `new_universe`'s uneven field only.
- **Small and quick**: defaults are grids of 56 to 64 boxes a side. Tests run every scene on
  4 or 6 cells of 4 boxes and check only that the numbers are finite.

## Measured (2026-11-15, defaults, all rules on, cell engine)

Expansion (8^3 cells of 6^3, 60 steps): occupied x1.55, radius 29.8 to 34.5, energy radius
23.7 to 22.5, drift 0, edge share 0.006. Hubble gap: late / early 0.0007 (known 1.09): growth
stops at the grid's edge. Structure: contrast 1.97 to 2.8, lumps 0 to 0. Bound lump: no lump
forms. Orbit: the small blob falls in at distance 10 (early period 62, Newton 62.5) and turns
0.6 of a turn at 14 (early period 34, Newton 104). Black hole: no ball seals (core energy
about 100 to 155 of 250 to 2000; one collapse at 2000).
Isotropy (2026-11-16): pull differs 2.2% by direction; flow is unreadable (carried 3.3 / -7.3 /
-7.5 axis / face / body, expected shift 0.004 boxes, rest scatter 0.06). Spin: winding lost,
strength 0.86 to 0.15, tilt 90, drift +0.15, into the joint.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes          # skips GPU scenes without CUDA
PYTHONPATH=. .venv/bin/python -c "from physics.scenes import orbit; print(orbit()['kepler'])"
```
