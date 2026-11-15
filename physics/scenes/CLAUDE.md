# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`, "How results are trusted"): each
scene builds a start, runs the fluid engine and returns measured numbers. Module map in
`__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler's 4 pi^2 / GM, the Hubble 73.5 / 67.4). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better; report it.
- **Only the engine's public face**: `new_grid`, `step`, `stats`, `Settings`, and
  `Grid.from_host` for placed blobs (`blob_start.py`); energies read from `grid.a_energy`
  and `grid.b_energy`. The engine's rules change under these scenes; its internals must not
  be reached into.
- **All rules on.** Scenes take `settings` (default `Settings.all_on()`); a rule switched off
  is only for a side-by-side run.
- **Box units.** The engine's boxes are fixed and no length follows energy yet (rule 4), so
  every size here is a box count or a distance in boxes: it says where the energy is, not
  that space grew.
- **Placed blobs** (orbit, black hole) break "nothing placed by hand" on purpose: those two
  test one mechanism. The expansion, structure, bound lump and Hubble scenes start from
  `new_grid`'s uneven field only.
- **Small and quick**: defaults are side 48 to 64 and under about 2 minutes each on the
  RTX 5080. Tests run every scene on side 16 and check only that the numbers are finite.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes          # skips GPU scenes without CUDA
PYTHONPATH=. .venv/bin/python -c "from physics.scenes import orbit; print(orbit()['kepler'])"
```
