"""Engine run: a universe of physics.engine, built and ready to be timed.

Contract: `engine_run(cells_per_side, seed=SEED, spare=SPARE, device=DEVICE)` builds the
engine's `new_universe` with the default `Settings` (every rule on) and returns a `Run`:
`advance()` is one engine `step`, `cells` is cells_per_side cubed, and `facts()` returns
  energy_volumes, joints  the two kinds counted at the start
  build_seconds           wall time of the build, synchronized
  steps_done              the universe's step counter now
  ids_used                cell ids handed out so far (the start's, plus one per torn piece)
  bytes_per_cell          GPU memory Warp's pool holds now, over the start cells
The pool is the whole process's: with another universe alive, bytes_per_cell counts it too.
This is the only file of the bench that imports physics.engine.
"""
from __future__ import annotations

import time

import warp as wp

from physics.bench.throughput.definitions.engine_start import DEVICE, SEED, SPARE
from physics.bench.throughput.definitions.run import Run
from physics.engine import ENERGY_VOLUME, JOINT, Settings, new_universe, step


def engine_run(cells_per_side: int, seed: int = SEED, spare: float = SPARE, device: str = DEVICE) -> Run:
    """A default engine universe of `cells_per_side` cells a side, as a Run."""
    settings = Settings()
    started = time.perf_counter()
    state = new_universe(settings, cells_per_side, seed, spare, device)
    wp.synchronize()
    build_seconds = time.perf_counter() - started
    kind = state.host("kind")
    cells = cells_per_side ** 3
    start = {"energy_volumes": int((kind == ENERGY_VOLUME).sum()), "joints": int((kind == JOINT).sum()),
             "build_seconds": build_seconds}

    def facts() -> dict:
        return {**start, "steps_done": state.step, "ids_used": state.count,
                "bytes_per_cell": wp.get_mempool_used_mem_current(wp.get_device(device)) / cells}
    return Run(lambda: step(state, settings), cells, facts)
