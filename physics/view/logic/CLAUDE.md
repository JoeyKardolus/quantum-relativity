# View, logic

`tracers.py` (the dots the flow carries), `streaks.py` (their tails as what the card draws), `swirl_painter.py` (OpenGL: the buffer, `load`, `draw`), `picture.py` (one drawing read back, for files and tests), `cell_window.py` (window and keys), `window.py` (the door and `--pictures`), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes (the engine's `before_pass`), by that pass's own pair rotation. Moved once a step instead, they drift into the core (measured).
- `COUNT`, `MARKS`, `MARK_EVERY`, `BOXES`, `SPEED`, `FRAMES` and `TICKS` only draw. Set by eye once; do not move them.
- No kernel argument may be a C++ word (`new`): the CPU build fails.
- pyglet is imported only when a window opens. Under WSL2 the two Mesa settings in `window.py` come before it; Warp and OpenGL then share no memory, so `load` copies by the host (4 MB a tick, straight into pyglet's own copy of the buffer).
- Never call `.numpy()`, `peek` or `boxes_of` in a frame, and in a tick only what must be: each waits for the card. `boxes_of` on a calm cell deals all its boxes on one thread (8 ms); the window asks once and keeps the answer.
- The title is worked out twice a second and costs 12 ms (`swirl_of`), one late frame each time.
- The picture tests skip where no window can open.
