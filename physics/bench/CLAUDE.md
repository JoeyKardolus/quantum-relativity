# physics/bench/

The bench (issue #3, the SIZE goal): how fast the engine steps and the largest universe
that runs at least 1 step a second. Module map in `__init__.py`.

## Must not get wrong

- **No engine rules in the library.** `throughput` and `largest_universe` time any
  `advance` callable; only `__main__.py` reaches `physics.engine`.
- **Size is counted in cells** (lattice places cubed). Each cell id carries 216 compute
  boxes, so memory goes by cells: `BYTES_PER_CELL` (1,770, measured by the engine's build).
- **Timing is synchronized** at both ends (`throughput`); warmup covers kernel compiles and
  a first step's set-up. Report every number with its sizes, counts and seed; never tune.
- **Size search** bisects an ascending list assuming bigger is never faster; a `MemoryError`
  from build means "does not fit", raised before allocating, so a probe never spills into
  WSL's shared system memory.
- **GPU one job at a time.** Other programs on the card change every number.
- **Gone**: universes side by side (`QuantaBatch`, the diffusion step). They stacked boxes
  as space on a batch axis; the cell-grid engine runs one universe per State.

## Measured (2026-11-17, RTX 5080, seed 1, defaults)

By the engine's build, on a card another job held 9 to 12 GB of: 1.0 million cells 21.3
steps/s, 2.1 million 10.6; about 22 million cell-steps a second; 8 million cells would
fit a free 16 GB card (calculated, not run). With this command, the same day, card
still shared: 32 places 48 steps/s over the first 20 steps; 64 places (262,144 cells) 2.6
over the first 3. 2,280 bytes a place held (1,824 a cell id with the quarter spare).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/bench
PYTHONPATH=. .venv/bin/python -m physics.bench throughput --cells 32
PYTHONPATH=. .venv/bin/python -m physics.bench size --sizes 32 64 128   # these sizes, no search
```
