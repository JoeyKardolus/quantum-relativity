# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each scene builds a start, runs
the engine and returns measured numbers. Module map in `__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value where
  there is one (Kepler, the Hubble 73.5 / 67.4, q about -0.55). No pass/fail, and never
  change a default, seed or start to make a scene look better.
- **Cells only.** A scene reads quanta, lengths, clocks and who neighbours whom, through
  the engine's public face. Planck boxes are not space: no scene reads a box, and there
  is no position, margin or empty space to measure in.
- **Size is the sum of cell lengths** (`stats()["size"]`); the universe wraps. Distance is
  counted in volumes along the lattice or in summed lengths, never in boxes.
- **A cell id is the same cell for good**, and `state.clock` is its own elapsed time. A
  volume that fell into another is gone; a tear's new cells have no lattice place.
- **All rules on.** A rule switched off is only for a side-by-side run.
- **Placed starts** (orbit, black hole, two lumps) break "nothing placed by hand" on
  purpose, through `packed_start`: extra energy in volumes of an even grid, books
  restarted (`start` passes fitted start amounts on).
- **Two lumps reads the pull beside its known law**: the same measurement on a
  1 / distance potential in the same wrapping box (`newton`), never a bare power; the
  lump alone is subtracted, because the even grid's own jitter tilts the pull too.
- **The orbit cannot be given a sideways speed**: crossing takes all direction off energy.
- **Small and quick**: 16 places, 200 steps, under a second each (two lumps: 48 places,
  minutes). Tests run every scene small and check only that the numbers are finite.

## Measured (2026-11-17, seed 1, defaults, no pass marks written)

Expansion: size 8,754.3 to 8,745.2 in 200 steps (x0.9990). Hubble late/early -0.91 (known
1.09). Structure: contrast 0.100 to 0.092, 0 lumps. Bound lump: the fullest volume holds
1.38 times the mean (no lump). Orbit: the blob keeps 85% and 89% of its extra energy at
200 steps, half is gone by step 400. Black hole: 4 and
8 times the mean do not seal; 16 times seals its six joints, no collapse. Timescape:
dense clocks 0.857 of the emptiest; q is meaningless (the size hardly moves). Dimension:
2.84 ball, 2.75 spectral, the plain grid's own numbers. Two lumps (48 places, 4 times
packed, 2, 3, 4, 6, 8 volumes apart, 2,000 steps): with crossing off the pull falls
1, 0.39, 0.21, 0.084, 0.040 beside 1, 0.27, 0.13, 0.048, 0.024 for one over distance
squared; from 3 volumes on its power is -2.00 beside -2.10. Every rule on: the lumps'
extra energy moves 0.28, 0.18, 0.35, 0.22, 0.04 volumes toward each other (alone: 0.02
at most) while 88% of it spreads away.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
```
