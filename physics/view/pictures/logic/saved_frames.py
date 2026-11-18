"""Saved frames: step a source and save a picture of its grid every few steps.

Contract: `saved_frames(source, steps, every, directory, picture)` saves
`picture(pieces, low, high)` (a uint8 (height, width, 3) RGB image of the source's
grid; `source` fits the `Source` interface) at step 0 and after every `every` steps up
to `steps`, as `<directory>/step_<00000>.png`, and returns one dict per picture: step,
path, bytes, and the source's stats(). `picture` is `WholePicture` (the whole grid in
three dimensions, needs a GPU) or `slice_of_the_middle`. Every picture is handed the
same `low` and `high`, the start's grid with a margin of MARGIN of its width all round.
Raises ValueError when `every` is below 1.
"""
from __future__ import annotations

from pathlib import Path

from physics.view.pictures.definitions.picture_marks import MARGIN
from physics.view.pictures.logic.write_png import write_png


def saved_frames(source, steps: int, every: int, directory, picture) -> list:
    """Run `steps` steps and save a picture every `every`."""
    if every < 1:
        raise ValueError(f"every: {every}, expected at least 1")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    frames, low, high = [], None, None
    while True:
        pieces = source.view()
        if low is None:
            pad = MARGIN * (pieces.high - pieces.low)
            low, high = pieces.low - pad, pieces.high + pad
        path = folder / f"step_{source.steps:05d}.png"
        size = write_png(path, picture(pieces, low, high))
        frames.append({"step": source.steps, "path": str(path), "bytes": size, **source.stats()})
        if source.steps >= steps:
            return frames
        for _ in range(min(every, steps - source.steps)):
            source.step()
