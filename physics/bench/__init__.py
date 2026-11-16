"""Bench: run many small universes side by side in one GPU job, and measure size.

Written against one tiny interface (step_function.py): a step over a QuantaBatch, each
Planck box one signed int32 whole number (bible rule 4) and one float32 phase, many
universes stacked on a leading batch dimension. It knows nothing of any engine's rules,
so it serves the rebuilt core (issue #3) unchanged; physics.engine is reached only by the
command line.

Module map:
  quanta_batch.py      QuantaBatch: (universes, side, side, side) int32 quanta + float32 phase, steps
  step_function.py     StepFunction: the interface, step(batch) in place, local, amounts exact
  new_quanta_batch.py  new_quanta_batch: a seeded batch filled on the GPU, MemoryError if it won't fit
  diffusion_step.py    diffusion_step: the trivial local step (same-sign pairs share), tests the harness
  amounts.py           amounts: each universe's plus and minus amount, exact int64
  throughput.py        throughput: universe-steps a second of any advance callable
  largest_universe.py  largest_universe: bisect sizes for the biggest at >= 1 step a second
  __main__.py          python -m physics.bench throughput|size diffusion|engine
"""
from physics.bench.quanta_batch import QuantaBatch
from physics.bench.step_function import StepFunction
from physics.bench.new_quanta_batch import new_quanta_batch
from physics.bench.diffusion_step import diffusion_step
from physics.bench.amounts import amounts
from physics.bench.throughput import throughput
from physics.bench.largest_universe import largest_universe

__all__ = ["QuantaBatch", "StepFunction", "new_quanta_batch", "diffusion_step", "amounts", "throughput",
           "largest_universe"]
