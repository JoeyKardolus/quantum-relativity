"""Run the fit search: python -m physics.fit OUT_DIR [--cells N --steps N --every N --points N --run-points N --workers N --minutes M].

Writes OUT_DIR/results.jsonl (one line per run, resumable) and OUT_DIR/summary.txt, and prints
one progress line per run and the summary's path at the end. OUT_DIR must be outside the repo
(tmp or object storage): results are data, not code.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from physics.fit.fit_search import fit_search
from physics.fit.summarize_search import summarize_search


def _parse(argv):
    parser = argparse.ArgumentParser(prog="python -m physics.fit", description="the fit search on expansion shape and structure")
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--cells", type=int, default=24, help="lattice places a side (24: 1,728 energy volumes, 13,824 cells)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--points", type=int, default=3000, help="Latin-hypercube points")
    parser.add_argument("--run-points", type=int, default=None, help="run only the first so many of the points")
    parser.add_argument("--minutes", type=float, default=None, help="start no new run after so many minutes; run again to resume")
    parser.add_argument("--workers", type=int, default=12, help="universes side by side, one process each")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse(argv)
    records = fit_search(arguments.out_dir / "results.jsonl", cells=arguments.cells, steps=arguments.steps,
                         every=arguments.every, random_points=arguments.points, random_run=arguments.run_points,
                         workers=arguments.workers, minutes=arguments.minutes)
    summary = arguments.out_dir / "summary.txt"
    summary.write_text(summarize_search(records))
    print(f"{len(records)} runs, {sum('error' in record for record in records)} errors; summary {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
