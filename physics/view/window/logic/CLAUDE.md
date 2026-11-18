# physics/view/window/logic/

What happens while the window is open.

- `use_gpu_driver.py`: sets the two Mesa variables unless the user set them.
- `title_line.py`: the title from a snapshot: step, counts, view, rates, state.
- `stepper.py`: `Stepper`: a daemon thread that steps the source and publishes
  snapshots; `latest()` hands each out once.
- `cell_window.py`: `CellWindow`: uploads the newest snapshot, draws it through the
  orbit camera, takes keys and mouse. Needs OpenGL.

## Must not get wrong

- `Stepper` methods are safe from any thread, but only the stepper thread touches the
  source.
- Stats are read at most every `STATS_EVERY` seconds, and at once after a reset, a
  pause or a change of view.
- The view is fitted on the first snapshot, after a reset, on F, and when the grid
  outgrows the last fit. Turning is kept.
- Closing the window stops the stepper.
- `tests/` checks the title against one exact line, and the stepper on a plain-array
  source, including a source that raises.
