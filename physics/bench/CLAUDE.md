# physics/bench/

The bench (issue #3, the SIZE goal): how fast the engine steps, and the largest universe
it still runs at one step a second. One directory per purpose, each with `definitions/`
(what things are), `logic/` (what happens) and `tests/`; `__main__.py` is the command.

## Must not get wrong

- **The bench holds no engine rule.** It times any callable that takes a universe one
  step. One file, `throughput/logic/engine_run.py`, imports `physics.engine`.
- **Size is counted in cells** at the start: cells a side, cubed. Tears and collapses
  change the count during a run; every rate is quoted on the start count.
- **One universe at a time.** The engine steps one universe per `Grid`, so there is no
  count of universes side by side any more.
- **Never tune.** The seed, the spare share, the step counts, the pass mark and the size
  ladder are written in `definitions/` before any run. Report every number with its
  size, steps and seed, and say what else was on the GPU.
- **Other programs on the GPU change every number**, speed and free memory both.

## Measured (2026-11-18, RTX 5080, seed 1, defaults, rebuilt engine)

The card was shared all day (other jobs held 3 to 14.5 GB and up to 100% of it), and
the numbers moved with them. No figure here is the engine's speed on a free card.

- 16 a side (4,096 cells), 20 steps after 1 warm-up: 53, 31 and 28 steps a second in
  three runs; an earlier run of 5 steps on a fuller card gave 1.1.
- 32 a side (32,768 cells): 4.6 over 20 steps; 1.5 over 3; 0.39 on the fuller card.
- One search (`size`, 3 steps a probe): 34 a side 0.85, 18 8.7, 26 3.5, 30 2.1, 32 1.5;
  answer 32 a side, limit speed. A probe of 3 steps is too short to trust on a shared
  card.
- Memory: 3,046 to 3,150 bytes a cell id (3,808 to 3,938 a start cell).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/bench
PYTHONPATH=. .venv/bin/python -m physics.bench throughput --cells 16
PYTHONPATH=. .venv/bin/python -m physics.bench size
PYTHONPATH=. .venv/bin/python -m physics.bench size --sizes 16 24 32   # each alone, no search
```
