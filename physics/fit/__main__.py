"""Run the fit search: python -m physics.fit OUT_DIR [--cells N --boxes N --steps N --every N --points N].

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
    parser.add_argument("--cells", type=int, default=7)
    parser.add_argument("--boxes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--points", type=int, default=80, help="Latin-hypercube points")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse(argv)
    records = fit_search(arguments.out_dir / "results.jsonl", cells=arguments.cells, boxes=arguments.boxes,
                         steps=arguments.steps, every=arguments.every, random_points=arguments.points)
    summary = arguments.out_dir / "summary.txt"
    summary.write_text(summarize_search(records))
    print(f"{len(records)} runs, {sum('error' in record for record in records)} errors; summary {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
