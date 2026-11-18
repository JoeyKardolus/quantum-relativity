"""Bench command line: the engine's speed, and the largest universe at one step a second.

  python -m physics.bench throughput [--cells 16] [--steps 20]
  python -m physics.bench size                   (even sides 4 to 64, bisected)
  python -m physics.bench size --sizes 8 16 24   (each of these sizes alone, no search)

Prints one JSON line per result: a throughput's rates with the run's facts, or a
search with every probe and its limit. The universe is the engine's default one
(engine_start.py); a size that would not fit the GPU's free memory is never built.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Sequence

from physics.bench.largest_universe.definitions.pass_mark import STEPS as SIZE_STEPS
from physics.bench.largest_universe.definitions.size_ladder import SIZES
from physics.bench.largest_universe.logic.fitting_run import fitting_run
from physics.bench.largest_universe.logic.largest_universe import largest_universe
from physics.bench.throughput.definitions.timing import STEPS as THROUGHPUT_STEPS
from physics.bench.throughput.logic.throughput import throughput


def _throughput_line(cells_per_side: int, steps: int) -> dict:
    """One throughput of the engine at this size: its rates and the run's facts."""
    run = fitting_run(cells_per_side)
    rates = throughput(run.advance, run.cells, steps)
    return {"cells_per_side": cells_per_side, **dataclasses.asdict(rates), **run.facts()}


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse the command and print its results as JSON lines."""
    parser = argparse.ArgumentParser(prog="python -m physics.bench")
    parser.add_argument("measure", choices=["throughput", "size"])
    parser.add_argument("--cells", type=int, default=16, help="throughput: cells a side")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    asked = parser.parse_args(arguments)
    if asked.measure == "throughput":
        print(json.dumps(_throughput_line(asked.cells, asked.steps or THROUGHPUT_STEPS)), flush=True)
        return
    for sizes in ([[size] for size in sorted(asked.sizes)] if asked.sizes else [SIZES]):
        search = largest_universe(fitting_run, sizes, steps=asked.steps or SIZE_STEPS)
        print(json.dumps(dataclasses.asdict(search)), flush=True)


if __name__ == "__main__":
    main()
