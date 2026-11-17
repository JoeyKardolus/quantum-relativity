# physics/scenes/

The fit's scorecard (bible `physics/quantum-relativity`): each scene builds a start, runs
the engine and returns numbers. Module map in `__init__.py`.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value. Pass
  marks are written to a file before a scorecard run and judged outside the scenes;
  never change a default, seed or start to make a scene look better.
- **Cells only.** A scene reads quanta, lengths, clocks and who neighbours whom, through
  the engine; nothing is measured in boxes. Only `spin` reads a
  block: a whirl lives inside a cell.
- **Size is the sum of cell lengths**; the universe wraps. Distance is counted in volumes
  or in summed lengths.
- **All rules on.** A rule is off only in a side-by-side run (two lumps and isotropy
  read the pull with crossing off beside the full run).
- **Placed starts** (orbit, black hole, two lumps, isotropy) break "nothing placed by
  hand" on purpose, through `packed_start`. Every scene takes `start`, a fit candidate's
  start amounts.
- **A pull is read beside its law in the same wrapping box**, never bare: the box is a
  cube and its copies pull too (`two_lumps` `newton`, `isotropy` `_box_law`). The twin
  run without the lump is subtracted: the even grid's jitter tilts the pull.
- **A slow pull has not arrived**: at light speed 0.1 to 0.3 (the fit's candidates) a
  2,000-step placed scene reads the pull before it settles.
- **No sideways speed** (crossing takes all direction off energy), **no belt trick** (no
  rule carries phase from cell to cell), and **quantum is staged** (`quantum_split`).
- No scene has an end rule: in a tear chain it takes seconds a step.

## Measured (2026-11-17, seed 1; default = fitted, as nothing is fitted)

Marks written first; 24 places, 5,000 steps, hold 20,000.
**2 of 12 pass**, the same on seeds 2 and 3. Misses: orbit 0.011 of a turn; bound lump: fullest volume 1.03 times the mean; black hole: 8, 16, 32 times
seal one volume each, radius goes as energy to the power -1.00 (known +1), 64 times puts
all 512 volumes in one cell; Hubble 0.02, of two shrinking rates (known 1.09); timescape:
lump clocks 0.971 of the emptiest's, nothing grows; expansion x0.9942; structure:
contrast 0.101 to 0.014; isotropy: pull off / on a row 0.90 to 1.20, arrived energy 0.84 to 1.29
(crossing off: pull within 0.5% of the box law); spin: one of four single whirls read 0 in one layer for a step (core at the
wall), a double whirl is gone by step 94; gravity: fall-off -1.94 beside -2.10 passes,
but lumps keep 11 to 14% of their energy. Passes: dimension 2.87 and 2.94 (the plain
grid's own; no event) and quantum. Candidates: steadiest 3 of 12 (structure by the letter, its universe x39 and
tearing at 20,000); best short 1 of 12, over by step 4,934.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes
```
