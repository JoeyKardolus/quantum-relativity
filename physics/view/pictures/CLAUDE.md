# physics/view/pictures/

Saved pictures: PNG files of the grid while a source steps, with no window shown.

- `definitions/`: a picture's fixed numbers and what a PNG chunk is.
- `logic/`: paints a slice on the CPU, draws the whole grid on the GPU off screen,
  writes PNG files, steps and saves.
- `tests/`: the GPU draws cubes whose picture is known.

## Must not get wrong

- **Every picture of a run shows the same box**: the start's grid plus a margin. So a
  cell drawn elsewhere later has really moved, and a grid that outgrows the box runs
  off the picture. That is the run, not a bug.
- **The whole grid needs a display and the GPU** (a hidden window); call
  `use_gpu_driver()` before importing `logic/whole_picture.py`. `--slice` needs neither.
- **PNG is written with the standard library only.** No image dependency.
- A machine keeps no datasets: pictures go to a directory outside the repo.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 40 --every 4 [--slice]
```
