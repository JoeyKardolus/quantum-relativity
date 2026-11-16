# physics/bench/

The bench (issue #3, the SIZE goal): many small universes side by side in one GPU job for the
fit, and the largest universe that runs at least 1 step a second. Module map in `__init__.py`.

## Must not get wrong

- **One interface, no engine rules.** The harness sees only a `QuantaBatch` (int32 signed
  quanta + float32 phase per Planck box, shape `(universes, side, side, side)`) and a step
  `step(batch)` that advances all universes in place and adds 1 to `batch.steps`. Never
  import an engine into the library modules; only `__main__.py` reaches `physics.engine`
  (which batches its own State the same way, stacked along the first axis).
- **Batch dimension, not separators.** Zero quanta is no space and space grows into it, so an
  empty separator between stacked universes would be claimed; the leading batch axis keeps
  every neighbour lookup inside its universe. A step must stop at index 0 and side - 1.
- **Amounts exact.** A step keeps each universe's plus and minus amount to the quantum
  (`amounts`); only joint collapse turns minus into plus. `diffusion_step` is a harness test,
  not physics: never use it in a scene or fit.
- **Timing is synchronized** at both ends (`throughput`); warmup covers kernel compiles and a
  first step's set-up. Report every number with its sizes, counts and seed; never tune.
- **Size search** bisects an ascending list assuming bigger is never faster; a `MemoryError`
  from build means "does not fit". `new_quanta_batch` raises it before allocating, so a probe
  never spills into WSL's shared system memory.
- **GPU one job at a time.**

## Measured (2026-11-16, RTX 5080, seed 1, defaults, 3 timed steps from the start)

Largest at >= 1 step/s, cells of 10 boxes, margin the filled width: ladder off 28^3 cells,
593M boxes, 2,197 energy volumes, 5.5 steps/s (29^3, 663M: 9.0 in one search, 0.19 in the
next: the edge of the 16 GB); ladder on 20^3, 216M boxes, 729 volumes, 2.2 steps/s
(21^3: 0.97), the ladder counting 0.31 to 0.36 bytes a box. The cell engine: 28.1M boxes,
64 volumes, 1.78 steps/s.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/bench          # GPU tests skip without CUDA
PYTHONPATH=. .venv/bin/python -m physics.bench throughput diffusion   # or engine; size diffusion|engine
PYTHONPATH=. .venv/bin/python -m physics.bench size engine --ladder   # the same with the ladder attached
```
