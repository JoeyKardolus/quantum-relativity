"""Step function: the one interface the bench harness asks of an engine's step.

Contract: a StepFunction is any callable `step(batch: QuantaBatch) -> None` that advances
every universe in the batch by one step, in place, and adds 1 to `batch.steps`. It may
read and write only a box and its neighbours within the same universe (the theory's
locality: every box only needs its neighbours), never couple two universes, and must keep
each universe's plus amount and minus amount exactly (bible rule 4; only joint collapse
turns minus quanta into plus quanta, keeping the total amount ignoring sign). The
harness checks the amounts with `amounts` and times steps with `throughput`; it never
looks inside the step. `diffusion_step` is the trivial example that tests the harness.
"""
from __future__ import annotations

from typing import Protocol

from physics.bench.quanta_batch import QuantaBatch


class StepFunction(Protocol):
    """Advance every universe of a batch by one step, in place."""

    def __call__(self, batch: QuantaBatch) -> None:
        """One step for all universes; batch.steps grows by 1."""
