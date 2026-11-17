"""Slice frames: step a source and save a picture of the middle slice every few steps, without a window.

Contract: `slice_frames(source, steps, every, directory, width=1100, layer=None)` saves
`slice_picture` of the source's grid (source interface: `cells_source.py`) at step 0
and after every `every` steps up to `steps`, as `<directory>/step_<00000>.png`, and
returns one dict per picture: step, path, bytes, and the source's stats(). `layer` is
the lattice index along z of the slice (None: `middle_layer`, the even layer nearest
the middle, where energy volumes lie). Every picture shows the same rectangle, the
start's grid with a margin of MARGIN of its width all round, so a cell that is drawn
elsewhere in a later picture has really moved.
"""
from __future__ import annotations

from pathlib import Path

from physics.view.middle_layer import middle_layer
from physics.view.slice_picture import slice_picture
from physics.view.write_png import write_png

MARGIN = 0.04


def slice_frames(source, steps: int, every: int, directory, width: int = 1100, layer: int | None = None) -> list:
    """Run `steps` steps and save a slice picture every `every`."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    frames, low, high = [], None, None
    while True:
        pieces = source.view()
        if low is None:
            pad = MARGIN * (pieces.high - pieces.low)
            low, high = pieces.low - pad, pieces.high + pad
        cut = middle_layer(pieces.side) if layer is None else layer
        path = folder / f"step_{source.steps:05d}.png"
        size = write_png(path, slice_picture(pieces, cut, low, high, width))
        frames.append({"step": source.steps, "path": str(path), "bytes": size, **source.stats()})
        if source.steps >= steps:
            return frames
        for _ in range(min(every, steps - source.steps)):
            source.step()
