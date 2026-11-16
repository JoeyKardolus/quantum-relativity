"""Bench command line: side-by-side throughput and the largest universe at 1 step a second.

  python -m physics.bench throughput diffusion [--side 32] [--counts 1 8 32 64] [--steps 50]
  python -m physics.bench throughput engine    [--counts 1 8 32 64] [--steps 20]
  python -m physics.bench size diffusion       (sides 32..1280 in steps of 32)
  python -m physics.bench size engine          (lattice places a side 8, 16, .. 160, 10 boxes a volume side)
  python -m physics.bench size engine --sizes 16 64 128   (these sizes only, rising, no search)
  python -m physics.bench size engine --ladder (the same with the detail ladder attached)

Prints one JSON line per result. `diffusion` runs a whole QuantaBatch per step call;
`engine` is physics.engine, all rules on: its throughput batches `count` universes side
by side in one State (3 cells a side of 10 boxes with a 1-box margin, a 32**3 grid each,
seed 1 + universe index) and steps them with one step call; its size search uses
new_universe's defaults (seed 1, margin 0) at lattice sizes that are multiples of 8, so
every probe wraps round and its side divides by 4 (the box exchange then runs 64 phases
at every size, and a bigger universe is never faster). It reports the largest size that
runs at least 1 step a second and, per probe: boxes, energy volumes, joints, steps a
second over `--steps` timed steps from the start (default 3, after 1 warm-up step), the
seconds the start took to build, `held_bytes_per_box` (the state's per-box arrays
between steps, from their sizes) and `peak_bytes_per_box` (the most GPU memory Warp's
pool held up to the end of the probe, over its boxes: the build and a step's scratch
included; None when an earlier, bigger probe set the mark). A size probe raises
MemoryError before building when its boxes would take more than the GPU's free memory
at 30 bytes a box (25 kept: quanta, phase, pull, pull rate, clock and owner 4 each,
marks 1; 4 for the crossing's arrivals each step; 1 for the cell tables and lists), so
no probe spills into WSL's shared system memory; other programs on the GPU shrink what
is free, and the search then stops at memory, not at speed (the result says which:
"limit"). With --ladder the engine's step also writes and switches the detail ladder
kept beside its boxes (physics.engine.attach_ladder); the probe then budgets 39 bytes a
box (the engine's 30, the 8 of the per-step block copy, about 1 of ladder rows) and
reports, after the timed steps, the ladder's bytes a box at each level, in use and
allocated, and its block counts (physics.ladder.memory_per_box) as "ladder".
"""
from __future__ import annotations

import argparse
import json

from physics.bench.diffusion_step import diffusion_step
from physics.bench.largest_universe import largest_universe
from physics.bench.new_quanta_batch import new_quanta_batch
from physics.bench.throughput import throughput

_ENGINE_BYTES_PER_BOX = 30
_LADDER_BYTES_PER_BOX = 39
_ENGINE_SIZES = list(range(8, 161, 8))


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


def _engine_build(cells_per_side: int, ladder: bool = False, ladder_bytes: list | None = None,
                  measured: dict | None = None):
    import time

    import warp as wp
    from physics.engine import Settings, attach_ladder, new_universe, step
    wrap = cells_per_side % 2 == 0 and (cells_per_side // 2 * 11) % 2 == 0
    side = cells_per_side // 2 * 11 if wrap else (cells_per_side - 1) // 2 * 11 + 1
    per_box = _LADDER_BYTES_PER_BOX if ladder else _ENGINE_BYTES_PER_BOX
    device = wp.get_device("cuda:0")
    if side ** 3 * per_box > device.free_memory:
        raise MemoryError(f"{side ** 3} boxes do not fit in the GPU's free memory")
    settings = Settings()
    before = wp.get_mempool_used_mem_high(device)
    started = time.perf_counter()
    state = new_universe(cells_per_side, 10)
    wp.synchronize()
    held = sum(getattr(state, name).capacity for name in ("quanta", "phase", "pull", "pull_rate", "clock", "owner",
                                                           "marks"))
    facts = {"energy_volumes": int((state.kind == 1).sum()), "joints": int((state.kind == -1).sum()),
             "build_seconds": time.perf_counter() - started, "held_bytes_per_box": held / state.side ** 3,
             "mark": before, "high": before}
    if measured is not None:
        measured[cells_per_side] = facts
    store = attach_ladder(state) if ladder else None

    def advance() -> None:
        step(state, settings)
        facts["high"] = wp.get_mempool_used_mem_high(device)
        if store is not None and ladder_bytes is not None:
            from physics.ladder import memory_per_box
            ladder_bytes.append({"size": cells_per_side, "step": state.step, **memory_per_box(store)})
    return advance, state.side ** 3


def _engine_sizes(sizes: list, ladder: bool, steps: int) -> dict:
    """Search `sizes` for the largest at 1 step a second, each probe with its cells, build time and bytes a box."""
    ladder_bytes: list = []
    measured: dict = {}

    def build(size: int):
        return _engine_build(size, ladder, ladder_bytes, measured)
    result = largest_universe(build, sizes, steps=steps)
    for probe in result["probes"]:
        facts = measured.get(probe["size"])
        if facts is not None:
            high = facts.pop("high")
            facts["peak_bytes_per_box"] = high / probe["boxes"] if high > facts.pop("mark") else None
            probe.update(facts)
        rows = [entry for entry in ladder_bytes if entry["size"] == probe["size"]]
        if rows:
            probe["ladder"] = rows[-1]
    slow = [probe for probe in result["probes"] if probe["fits"] and probe["steps_per_second"] < 1.0]
    result["limit"] = "speed" if slow else ("memory" if any(not probe["fits"] for probe in result["probes"])
                                            else "the size list")
    result["ladder"] = ladder
    return result


def main() -> None:
    """Parse the command and print its results as JSON lines."""
    parser = argparse.ArgumentParser(prog="python -m physics.bench")
    parser.add_argument("measure", choices=["throughput", "size"])
    parser.add_argument("engine", choices=["diffusion", "engine"])
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
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
        print(json.dumps({"engine": "diffusion", **largest_universe(_diffusion_build, list(range(32, 1281, 32)))}),
              flush=True)
        return
    if arguments.sizes:
        for size in sorted(arguments.sizes):
            print(json.dumps({"engine": "engine", **_engine_sizes([size], arguments.ladder, arguments.steps or 3)}),
                  flush=True)
        return
    print(json.dumps({"engine": "engine", **_engine_sizes(_ENGINE_SIZES, arguments.ladder, arguments.steps or 3)}),
          flush=True)


if __name__ == "__main__":
    main()
