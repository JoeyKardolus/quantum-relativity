"""Saved frames: step a source and save a picture of its grid every few steps.

Contract: `saved_frames(source, steps, every, directory, picture, whole=False)` saves a
picture (uint8 (height, width, 3) RGB; `source` fits the `Source` interface) at step 0
and after every `every` steps up to `steps`, as `<directory>/step_<00000>.png`, and
returns one dict per picture: step, path, bytes, and the source's stats(). The picture
is `picture(plane, low, high)` of the slice through the middle (`source.view(layer)`,
`middle_layer`; `slice_of_the_middle` draws it without a GPU), or with `whole`
`picture(source.view(), low, high, plane)`: the packed cells as blocks over that same
slice (`WholePicture`, needs a GPU). `low` and `high` are the grid's own frame at that
step with a margin of MARGIN of its width all round, so the frame fills every picture
alike and a cell drawn elsewhere has moved within the grid. Raises ValueError when
`every` is below 1.
"""
from __future__ import annotations

from pathlib import Path

from physics.view.drawing.definitions.middle_layer import middle_layer
from physics.view.pictures.definitions.picture_marks import MARGIN
from physics.view.pictures.logic.write_png import write_png


def saved_frames(source, steps: int, every: int, directory, picture, whole: bool = False) -> list:
    """Run `steps` steps and save a picture every `every`."""
    if every < 1:
        raise ValueError(f"every: {every}, expected at least 1")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    layer = middle_layer(source.view(0).side)
    frames = []
    while True:
        plane = source.view(layer)
        pad = MARGIN * (plane.high - plane.low)
        low, high = plane.low - pad, plane.high + pad
        path = folder / f"step_{source.steps:05d}.png"
        size = write_png(path, picture(source.view(), low, high, plane) if whole else picture(plane, low, high))
        frames.append({"step": source.steps, "path": str(path), "bytes": size, **source.stats()})
        if source.steps >= steps:
            return frames
        for _ in range(min(every, steps - source.steps)):
            source.step()
