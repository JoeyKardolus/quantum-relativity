# View, logic

`paint.py` (the red glow), `tracers.py` (the dots the flow carries), `draw_tracers.py`, `cell_window.py` (window and keys), `window.py` (the door and `--pictures`), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes (the engine's `before_pass`), by that pass's own pair rotation. Moved once a step instead, they drift into the core (measured).
- `GLOW`, `SAMPLES`, `COUNT`, `MARKS`, `LIGHT`, `HEAD`, `TAIL`, `PIECES`, `BOXES` and `SPEED` only draw. Set by eye once; do not move them.
- No kernel argument may be a C++ word (`new`): the CPU build fails.
- pyglet is imported only when a window opens. Under WSL2 the two Mesa settings in `window.py` come before it.
