# View, logic

`tracers.py`, `tracers_across.py` (dots the flow carries), `streaks.py`, `line_streaks.py` (their tails), `placed.py` (where each cell is drawn), `scene.py` (scenes, the cut), `swirl_painter.py`, `picture.py`, `cell_window.py` (keys), `window.py` (the door), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes; moved once a step, tracers drift into the core.
- A tracer's second number names its cell's slot, `across` a slot.
- The cut's tracers are laid in the two middle layers of boxes of one layer's cells. Joints from layer to layer are in no cut.
- A streak the other way round the wrap is not drawn.
- Drawing numbers (`COUNT`, `MARKS`, `AT_MEAN`, `SPEED`, `TICKS`) were set by eye once; do not move them.
- No kernel argument may be a C++ word.
- Never call `.numpy()` in a frame. Under WSL2 `load` copies by the host.
