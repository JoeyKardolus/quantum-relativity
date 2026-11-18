# physics/measure/

Numbers read off the engine's grid. One directory per measure; each holds `definitions/`
(what things are), `logic/` (what happens), `tests/` and its own `CLAUDE.md`.

- `dimension/`: how many dimensions the grid of energy volumes has, read only from which
  volumes neighbour which.
- `winding/`: how many whole turns a cell's fluid makes round an axis of its own block.

`__init__.py` holds only the public names. `tests/` runs both measures on the real engine.

## Must not get wrong

- **Nothing loose.** A directory here holds subdirectories plus only `__init__.py`,
  `conftest.py` and `CLAUDE.md`. A new measure is a new directory of the same shape.
- **Proved first.** A measure is used on the engine only after a test gets a known answer
  out of it. A scene reports the reference's number beside the engine's.
- **Read, never change.** A measure takes host arrays the engine's public names hand out
  (`read_ways`, `read_cells`, `state.block`, `state.host`). It never steps the engine and
  never writes to it. The only engine import in the code is the constant `ENERGY_VOLUME`.
- **Boxes are not space.** No measure may walk from one cell's boxes into another's. The
  winding is read inside one block; the dimension reads relations only, no lengths.
- **Definitions never import logic.**
- **A failing test is a measure that is not proved.** Never bend one to pass.

## Known on the real engine (16 a side, seed 1)

- The start is exactly the wrapped 26-neighbour grid of 512 volumes, and it still is
  after the 24 steps of the held run (tested).
- Every cell of the start reads one whole turn its own way in all six layers (tested).

## Not done

- **The spectral dimension is not proved on the engine's shape.** Two tests are marked as
  strict expected failures; see `dimension/CLAUDE.md`.
- **Deleted, nothing used them:** `whirl_line`, `loop_circulation`, `whirl_strength`. They
  followed a swirl's line through boxes as if boxes were space; a swirl now lives inside
  one cell's block of 6 boxes a side.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/measure     # uses CUDA when there is one, else the CPU
```
