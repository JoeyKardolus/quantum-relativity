"""Command line: `python -m physics.view [--volumes N] [--seed S] [--unevenness U] [--set name=value ...] [--steps-per-frame K]`.

Contract: sets the Mesa GPU variables (`use_gpu_driver`) before pyglet is imported,
builds an `AbabSource` (all rules on, `--set` overrides of `Settings` fields, each a
float; an unknown name or a value that is not a number exits with a message),
steps it on a `Stepper` thread, K steps between drawn frames, and opens the
`PointWindow` until it is closed.
"""
from __future__ import annotations

import argparse

from physics.view.gpu_driver import use_gpu_driver


def main() -> None:
    """Parse the options, then step and draw until the window closes."""
    parser = argparse.ArgumentParser(prog="python -m physics.view")
    parser.add_argument("--volumes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unevenness", type=float, default=0.5)
    parser.add_argument("--set", action="append", default=[], metavar="name=value",
                        help="override one rule setting, e.g. --set leak=0.1")
    parser.add_argument("--steps-per-frame", type=int, default=1)
    arguments = parser.parse_args()
    overrides = {}
    for pair in arguments.set:
        name, separator, value = pair.partition("=")
        try:
            overrides[name.strip()] = float(value)
        except ValueError:
            parser.error(f"--set {pair}: expected name=number")
        if not separator:
            parser.error(f"--set {pair}: expected name=number")
    use_gpu_driver()
    from physics.view.abab_source import AbabSource
    try:
        source = AbabSource(arguments.volumes, arguments.seed, arguments.unevenness, overrides)
    except TypeError as raised:
        parser.error(f"--set: {raised}")
    from physics.view.point_window import PointWindow
    from physics.view.stepper import Stepper
    PointWindow(Stepper(source, arguments.steps_per_frame)).run()


if __name__ == "__main__":
    main()
