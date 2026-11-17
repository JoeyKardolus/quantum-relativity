"""Command line: `python -m physics.view [--cells N] [--seed S] [--unevenness U] [--slope S] [--set name=value ...] [--steps-per-frame F] [--pictures DIR --steps N --every K [--slice]]`.

Contract: builds a `CellsSource` (physics.engine, the default universe: 32 lattice
places a side, seed 1; `--set` overrides of `Settings` fields, each a number; an unknown
name or a value that is not a number exits with a message). Without `--pictures` it sets
the Mesa GPU variables (`use_gpu_driver`) before pyglet is imported, steps the source on
a `Stepper` thread, F steps between drawn frames, starting paused (space starts), and
opens the `CellWindow` on the whole grid until it is closed. With `--pictures DIR` no
window shows: `saved_frames` runs `--steps` steps and saves a picture every `--every`
steps into DIR, printing one line per picture. The picture is the whole grid in three
dimensions, drawn by the GPU off screen (`WholePicture`), or with `--slice` the middle
slice, drawn without GPU or display.
"""
from __future__ import annotations

import argparse

from physics.view.gpu_driver import use_gpu_driver


def main() -> None:
    """Parse the options, then step and draw until the window closes, or save pictures."""
    parser = argparse.ArgumentParser(prog="python -m physics.view")
    parser.add_argument("--cells", type=int, default=32, help="lattice places along each axis (even)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unevenness", type=float, default=0.1,
                        help="root mean square of the log energy across the start's energy volumes")
    parser.add_argument("--slope", type=float, default=-3.0,
                        help="spectral slope of the start's ripples, P(k) = k**slope (-3: every size alike)")
    parser.add_argument("--set", action="append", default=[], metavar="name=value",
                        help="override one rule setting, e.g. --set crossing_rate=0.01")
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--pictures", metavar="DIR", help="save pictures of the whole grid here instead of opening a window")
    parser.add_argument("--slice", action="store_true", help="with --pictures: the middle slice instead of the whole grid")
    parser.add_argument("--steps", type=int, default=5000, help="with --pictures: steps to run")
    parser.add_argument("--every", type=int, default=250, help="with --pictures: steps between pictures")
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
    from physics.view.cells_source import CellsSource
    try:
        source = CellsSource(arguments.cells, arguments.seed, arguments.unevenness, arguments.slope, overrides)
    except TypeError as raised:
        parser.error(f"--set: {raised}")
    if arguments.pictures:
        from physics.view.saved_frames import saved_frames, slice_of_the_middle
        if arguments.slice:
            picture = slice_of_the_middle
        else:
            use_gpu_driver()
            from physics.view.whole_picture import WholePicture
            picture = WholePicture()
        for frame in saved_frames(source, arguments.steps, arguments.every, arguments.pictures, picture):
            print(f"step {frame['step']}: {frame['path']} size {frame['size']:.1f} lumps {frame['lumps']} "
                  f"collapses {frame['collapses']} tears {frame['tears']} quanta {frame['amount']}", flush=True)
        return
    use_gpu_driver()
    from physics.view.cell_window import CellWindow
    from physics.view.stepper import Stepper
    CellWindow(Stepper(source, arguments.steps_per_frame, paused=True)).run()


if __name__ == "__main__":
    main()
