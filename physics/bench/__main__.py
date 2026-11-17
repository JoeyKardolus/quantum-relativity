"""Bench command line: the engine's speed, and the largest universe at 1 step a second.

  python -m physics.bench throughput [--cells 32] [--steps 20]
  python -m physics.bench size                     (lattice places a side 16, 32, .. 256, bisected)
  python -m physics.bench size --sizes 32 100 128  (these sizes only, rising, no search)

Prints one JSON line per result. The universe is physics.engine's `new_universe(places,
seed=1)` with the default Settings, every rule on. A probe reports its cells (places
cubed), energy volumes, joints, steps a second over `--steps` timed steps from the
start (default 20 for throughput, 3 for size, after 1 warm-up step), the seconds the
start took to build and `bytes_per_cell`, the GPU memory Warp's pool held at the end
over the cells. A size probe raises MemoryError before building when its cell ids
(places cubed, plus the quarter kept spare for tears) at BYTES_PER_CELL would take
more than the GPU's free memory, so no probe spills into WSL's shared system memory;
other programs on the GPU shrink what is free, and the search then stops at memory,
not at speed (the result says which: "limit").
"""
from __future__ import annotations

import argparse
import json
import time

from physics.bench.largest_universe import largest_universe
from physics.bench.throughput import throughput

BYTES_PER_CELL = 1770
_SIZES = list(range(16, 257, 16))


def _build(places: int, measured: dict | None = None):
    """A default universe of `places` lattice places a side: (advance, cells); MemoryError when it will not fit."""
    import warp as wp
    from physics.engine import Settings, new_universe, step
    device = wp.get_device("cuda:0")
    if places ** 3 * 1.25 * BYTES_PER_CELL > device.free_memory:
        raise MemoryError(f"{places ** 3} cells do not fit in the GPU's free memory")
    settings = Settings()
    started = time.perf_counter()
    state = new_universe(places)
    wp.synchronize()
    facts = {"energy_volumes": int((state.kind_h == 1).sum()), "joints": int((state.kind_h == -1).sum()),
             "build_seconds": time.perf_counter() - started}
    if measured is not None:
        measured[places] = facts

    def advance() -> None:
        step(state, settings)
        facts["bytes_per_cell"] = wp.get_mempool_used_mem_current(device) / places ** 3
    return advance, places ** 3


def _sizes(sizes: list, steps: int) -> dict:
    """Search `sizes` for the largest at 1 step a second, each probe with its cells, build time and bytes a cell."""
    measured: dict = {}
    result = largest_universe(lambda size: _build(size, measured), sizes, steps=steps)
    for probe in result["probes"]:
        probe.update(measured.get(probe["size"], {}))
    slow = [probe for probe in result["probes"] if probe["fits"] and probe["steps_per_second"] < 1.0]
    result["limit"] = "speed" if slow else ("memory" if any(not probe["fits"] for probe in result["probes"])
                                            else "the size list")
    return result


def main() -> None:
    """Parse the command and print its results as JSON lines."""
    parser = argparse.ArgumentParser(prog="python -m physics.bench")
    parser.add_argument("measure", choices=["throughput", "size"])
    parser.add_argument("--cells", type=int, default=32, help="throughput: lattice places a side")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    arguments = parser.parse_args()
    if arguments.measure == "throughput":
        measured: dict = {}
        advance, cells = _build(arguments.cells, measured)
        result = throughput(advance, 1, cells, steps=arguments.steps or 20, warmup=1)
        print(json.dumps({"places": arguments.cells, **result, **measured[arguments.cells]}), flush=True)
        return
    for sizes in ([[size] for size in sorted(arguments.sizes)] if arguments.sizes else [_SIZES]):
        print(json.dumps(_sizes(sizes, arguments.steps or 3)), flush=True)


if __name__ == "__main__":
    main()
