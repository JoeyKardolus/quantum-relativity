"""Command line: `python -m physics.view [--cells N] [--boxes-per-cell K] [--seed S] [--unevenness U] [--slope S] [--set name=value ...] [--steps-per-frame F]`.

Contract: sets the Mesa GPU variables (`use_gpu_driver`) before pyglet is imported,
builds a `CellsSource` (physics.engine, a checkerboard of N^3 cells of K^3 Planck
boxes each inside an empty margin, `--set` overrides of `Settings` fields, each a
number (whole for a whole-number field); an unknown name or a value that is not a number exits with a message),
steps it on a `Stepper` thread, F steps between drawn frames, starting paused
(space starts), and opens the `PointWindow` until it is closed.
"""
from __future__ import annotations

import argparse

from physics.view.gpu_driver import use_gpu_driver


def main() -> None:
    """Parse the options, then step and draw until the window closes."""
    parser = argparse.ArgumentParser(prog="python -m physics.view")
    parser.add_argument("--cells", type=int, default=8, help="cells along each edge of the start's checkerboard")
    parser.add_argument("--boxes-per-cell", type=int, default=6, help="Planck boxes along each edge of a start cell")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unevenness", type=float, default=0.3,
                        help="root mean square of the log energy across the start")
    parser.add_argument("--slope", type=float, default=-3.0,
                        help="spectral slope of the start's ripples, P(k) = k**slope (-3: every size alike)")
    parser.add_argument("--set", action="append", default=[], metavar="name=value",
                        help="override one rule setting, e.g. --set friction=2")
    parser.add_argument("--steps-per-frame", type=int, default=1)
    arguments = parser.parse_args()
    overrides = {}
    for pair in arguments.set:
        name, separator, value = pair.partition("=")
        if not separator:
            parser.error(f"--set {pair}: expected name=number")
        try:
            overrides[name.strip()] = float(value)
        except ValueError:
            parser.error(f"--set {pair}: expected name=number")
    use_gpu_driver()
    from physics.view.cells_source import CellsSource
    try:
        source = CellsSource(arguments.cells, arguments.boxes_per_cell, arguments.seed,
                             arguments.unevenness, arguments.slope, overrides)
    except TypeError as raised:
        parser.error(f"--set: {raised}")
    from physics.view.point_window import PointWindow
    from physics.view.stepper import Stepper
    PointWindow(Stepper(source, arguments.steps_per_frame, paused=True)).run()


if __name__ == "__main__":
    main()
