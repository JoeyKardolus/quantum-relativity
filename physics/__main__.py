"""Command line: `python -m physics [--seed N] [--out DIR]`.

Contract: runs every milestone experiment with the seed, writes plots under
DIR/<experiment>/ and the results page to DIR/index.html, and prints one
summary line per milestone plus the page path.
"""
from __future__ import annotations

import argparse
import os

from physics.experiments import run_all
from physics.report import build_page


def main() -> None:
    """Run all experiments and write the results page."""
    parser = argparse.ArgumentParser(prog="python -m physics")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="/home/dupe/quantum-relativity-runs")
    arguments = parser.parse_args()
    results = run_all(arguments.seed, arguments.out)
    for result in results:
        print(f"{result['milestone']} {result['title']}: {result['verdict']} ({result['seconds']} s)")
    print(build_page(results, os.path.join(arguments.out, "index.html")))


if __name__ == "__main__":
    main()
