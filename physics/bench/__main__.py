"""Bench command line: side-by-side throughput and the largest universe at 1 step a second.

  python -m physics.bench throughput diffusion [--side 32] [--counts 1 8 32 64] [--steps 50]
  python -m physics.bench throughput engine    [--counts 1 8 32 64] [--steps 20]
  python -m physics.bench size diffusion       (sides 32..1280 in steps of 32)
  python -m physics.bench size engine          (cells a side 3..63, 10 boxes a cell side)
  python -m physics.bench size engine --ladder (the same with the detail ladder attached)

Prints one JSON line per result. `diffusion` runs a whole QuantaBatch per step call;
`engine` is physics.engine, all rules on: its throughput batches `count` universes side
by side in one State (3 cells a side of 10 boxes with a 1-box margin, a 32**3 grid each,
seed 1 + universe index) and steps them with one step call; its size search uses
new_universe's default margin (the filled cube's width) and reports energy volumes
beside boxes. A size probe raises MemoryError before building when its boxes would take
more than the GPU's free memory at 21 bytes a box (the engine's arrays), so no probe
spills into WSL's shared system memory. The cell engine (physics.potts) this replaced
measured 22 universe-steps a second at 32**3 and 28.1 million boxes (64 energy volumes)
at 1.78 steps a second (2026-11-16). With --ladder the engine's step also writes and
switches the detail ladder kept beside its boxes (physics.engine.attach_ladder); the probe
then budgets 30 bytes a box (the engine's 21, the 8 of the per-step block copy, about 1 of
ladder rows) and reports the ladder's own bytes a box after the timed steps
(physics.ladder.memory_per_box "in_use").
"""
from __future__ import annotations

import argparse
import json

from physics.bench.diffusion_step import diffusion_step
from physics.bench.largest_universe import largest_universe
from physics.bench.new_quanta_batch import new_quanta_batch
from physics.bench.throughput import throughput

_ENGINE_BYTES_PER_BOX = 21
_LADDER_BYTES_PER_BOX = 30


def _diffusion_throughput(count: int, side: int, steps: int) -> dict:
    batch = new_quanta_batch(count, side)
    return throughput(lambda: diffusion_step(batch), count, batch.boxes_per_universe, steps=steps, warmup=3)


def _engine_throughput(count: int, steps: int) -> dict:
    from physics.engine import Settings, new_universe, step
    settings = Settings()
    state = new_universe(3, 10, seed=1, margin=1, universes=count)
    return throughput(lambda: step(state, settings), count, state.side ** 3, steps=steps, warmup=1)


def _diffusion_build(side: int):
    batch = new_quanta_batch(1, side)
    return (lambda: diffusion_step(batch)), batch.boxes_per_universe


def _engine_build(cells_per_side: int, ladder: bool = False, ladder_bytes: list | None = None):
    import warp as wp
    from physics.engine import Settings, attach_ladder, new_universe, step
    filled = cells_per_side * 10
    side = -(-(3 * filled) // 8) * 8
    per_box = _LADDER_BYTES_PER_BOX if ladder else _ENGINE_BYTES_PER_BOX
    if side ** 3 * per_box > wp.get_device("cuda:0").free_memory:
        raise MemoryError(f"{side ** 3} boxes do not fit in the GPU's free memory")
    settings = Settings()
    state = new_universe(cells_per_side, 10)
    if not ladder:
        return (lambda: step(state, settings)), state.side ** 3
    from physics.ladder import memory_per_box
    store = attach_ladder(state)

    def advance() -> None:
        step(state, settings)
        if ladder_bytes is not None:
            ladder_bytes.append({"size": cells_per_side, "step": state.step,
                                 "in_use": memory_per_box(store)["in_use"]})
    return advance, state.side ** 3


def main() -> None:
    """Parse the command and print its results as JSON lines."""
    parser = argparse.ArgumentParser(prog="python -m physics.bench")
    parser.add_argument("measure", choices=["throughput", "size"])
    parser.add_argument("engine", choices=["diffusion", "engine"])
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ladder", action="store_true")
    arguments = parser.parse_args()
    if arguments.measure == "throughput":
        for count in arguments.counts:
            if arguments.engine == "diffusion":
                result = _diffusion_throughput(count, arguments.side, arguments.steps or 50)
            else:
                result = _engine_throughput(count, arguments.steps or 20)
            print(json.dumps({"engine": arguments.engine, **result}), flush=True)
        return
    if arguments.engine == "diffusion":
        result = largest_universe(_diffusion_build, list(range(32, 1281, 32)))
    else:
        ladder_bytes: list = []
        result = largest_universe(lambda size: _engine_build(size, arguments.ladder, ladder_bytes),
                                  list(range(3, 64)))
        for probe in result["probes"]:
            probe["energy_volumes"] = ((probe["size"] - 1) // 2) ** 3
            in_use = [entry["in_use"] for entry in ladder_bytes if entry["size"] == probe["size"]]
            if in_use:
                probe["ladder_bytes_per_box"] = in_use[-1]
        result["ladder"] = arguments.ladder
    print(json.dumps({"engine": arguments.engine, **result}), flush=True)


if __name__ == "__main__":
    main()
