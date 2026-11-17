# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each scene builds a start, runs
the engine (`physics.engine`) and returns measured numbers. Module map in `__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler, the Hubble 73.5 / 67.4, q about -0.55). No pass/fail, no threshold,
  and never change a default setting, seed or start to make a scene look better.
- **Cells only.** A scene reads quanta, lengths, clocks and who neighbours whom, through
  the engine's public face (`new_universe`, `step`, `stats`, `way_list`, `State.host`,
  `State.fill`, `cell_laws`, `totals`). Planck boxes are not space: no scene reads a box,
  and there is no position, margin or empty space to measure in.
- **Size is the sum of cell lengths** (`stats()["size"]`); the universe wraps. Distance
  between volumes is counted in volumes along the lattice or in summed lengths, never in
  boxes.
- **A cell id is the same cell for good**, and `state.clock` is its own elapsed time: follow
  a volume by its id. A volume that fell into another is gone (its quanta are in the
  other); a tear's new cells have no lattice place (`volume_field` reports their share).
- **All rules on.** Scenes take `settings` (default `Settings()`); a rule switched off is
  only for a side-by-side run.
- **Placed starts** (orbit, black hole) break "nothing placed by hand" on purpose, through
  `packed_start`: extra energy in volumes of an even grid, books restarted. The other
  scenes start from `new_universe`'s uneven field only.
- **The orbit cannot be given a sideways speed**: crossing takes all direction off energy.
  The scene measures how long the small blob holds and where its energy goes.
- **Small and quick**: defaults are 16 places (512 volumes), 200 steps, under a second
  each. Tests run every scene on 8 places and check only that the numbers are finite.

## Measured (2026-11-17, seed 1, defaults, no pass marks written)

Expansion: size 8,754.3 to 8,745.2 in 200 steps (x0.9990), rate trend -8e-6 then +4e-6.
Hubble late/early -0.91 (known 1.09). Structure: contrast 0.100 to 0.092, power x0.99 at
the largest size to x0.68 at the smallest, 0 lumps. Bound lump: the fullest volume holds
1.38 times the mean (no lump), length x1.006, the rest x0.998. Orbit: the blob keeps 85%
and 89% of its extra energy at 200 steps, half is gone by step 400, turned -0.012 and
0.006 of a turn. Black hole: 4 and 8 times the mean do not seal; 16 times seals its six
joints (a region of one volume), no collapse. Timescape: dense clocks 0.857 of the
emptiest; q 2,797 dense, 2,827 counter, 2,838 empty (the size hardly moves). Dimension:
2.84 ball, 2.75 spectral, the plain grid's own numbers.

Gone with boxes as space: isotropy, spin and belt trick (a whirl cannot cross a joint:
no fluid does), blob starts in open space, energy radius.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
```
