# physics/view/pictures/

Saved pictures: PNG files of the grid while a source steps, with no window shown.

- `definitions/`: a picture's fixed numbers and what a PNG chunk is.
- `logic/`: paints a slice on the CPU, draws the whole grid on the GPU off screen,
  writes PNG files, steps and saves.
- `tests/`: the GPU draws cubes whose picture is known.

## Must not get wrong

- **The slice is the default** and needs neither GPU nor display. It lays out one
  layer, so it costs the same at any lattice size.
- **Every picture is fitted to the grid's own frame** plus a margin. The frame is the
  perfect grid's, so the grid sits still from picture to picture and a cell drawn
  elsewhere has moved within it. The universe's size is in the printed line.
- **The whole grid needs a display and the GPU** (a hidden window); call
  `use_gpu_driver()` before importing `logic/whole_picture.py`.
- **PNG is written with the standard library only.** No image dependency.
- A machine keeps no datasets: pictures go to a directory outside the repo.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 40 --every 4 [--whole]
```
