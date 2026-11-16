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
  from build means "does not fit", raised before allocating (30 B a box for the engine), so
  a probe never spills into WSL's shared system memory.
- **GPU one job at a time.**

## Measured (2026-11-17, RTX 5080, seed 1, defaults, 5 timed steps after 1)

Engine, every rule on, wrapping, cells of 10 boxes (`size engine --sizes ...`; other
programs came and went on the card). Places: boxes, steps/s, build time:
16: 0.68M, 56.6, 0.4 s. 88: 113M, 1.95, 33 s. 96: 147M (110,592 volumes), 1.41, 43 s.
104: 187M, 0.12 (a collapse read the whole universe inside the timed steps).
112: 234M, 0.95, 67 s. 120: 287M (216,000 volumes), 0.74, 92 s.

Largest at 1 step a second: 96 places, 147M boxes. 25 B a box held, peak 29.4, or 34
when a collapse reads the whole universe; an empty card would hold 136 places (418M),
which did not run to the end beside the other programs. Speed limits it: box exchange,
mortar shift (strided launches) and per-cell rules in numpy on the host. From 64 places
up the start has joints past collapse_packing (120 places: 143 collapses in step 1,
89 s). Before: 3.65M boxes at 16 steps/s; 32 places never finished step 1.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/bench          # GPU tests skip without CUDA
PYTHONPATH=. .venv/bin/python -m physics.bench throughput diffusion   # or engine; size diffusion|engine
PYTHONPATH=. .venv/bin/python -m physics.bench size engine --sizes 16 96 120   # these sizes, no search
PYTHONPATH=. .venv/bin/python -m physics.bench size engine --ladder   # the same with the ladder attached
```
