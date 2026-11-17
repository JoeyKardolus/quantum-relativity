"""Run the fit search: python -m physics.fit OUT_DIR [--cells N --boxes N --steps N --every N --points N --workers N].

With --compare it runs compare_fits instead (gradient fit against random search on the
side-by-side stand-in, both checked on the real score) and writes OUT_DIR/summary.json.

Writes OUT_DIR/results.jsonl (one line per run, resumable) and OUT_DIR/summary.txt, and prints
one progress line per run and the summary's path at the end. OUT_DIR must be outside the repo
(tmp or object storage): results are data, not code.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from physics.fit.compare_fits import compare_fits
from physics.fit.fit_search import fit_search
from physics.fit.summarize_search import summarize_search


def _parse(argv):
    parser = argparse.ArgumentParser(prog="python -m physics.fit", description="the fit search on expansion shape and structure")
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--cells", type=int, default=24, help="start lattice places a side (24: 1,728 energy volumes)")
    parser.add_argument("--boxes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--points", type=int, default=100, help="Latin-hypercube points")
    parser.add_argument("--workers", type=int, default=8, help="universes side by side, one process each")
    parser.add_argument("--compare", action="store_true", help="gradient fit against random search")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse(argv)
    if arguments.compare:
        summary = compare_fits(arguments.out_dir)
        (arguments.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"compare written to {arguments.out_dir / 'summary.json'}")
        return 0
    records = fit_search(arguments.out_dir / "results.jsonl", cells=arguments.cells, boxes=arguments.boxes,
                         steps=arguments.steps, every=arguments.every, random_points=arguments.points,
                         workers=arguments.workers)
    summary = arguments.out_dir / "summary.txt"
    summary.write_text(summarize_search(records))
    print(f"{len(records)} runs, {sum('error' in record for record in records)} errors; summary {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
