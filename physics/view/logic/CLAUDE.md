# View, logic

`tracers.py`, `tracers_across.py` (dots the flow carries), `streaks.py`, `line_streaks.py` (their marks), `placed.py`, `scene.py` (scenes, cut, slab), `swirl_painter.py`, `picture.py`, `cell_window.py` (keys), `window.py` (the door), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes; moved once a step, tracers drift into the core.
- A tracer's second number names its slot, `across` a slot, in one float: keep it 0.001 off the slot's far edge.
- The cut: the two middle layers of boxes of one layer's cells. A slab adds the joints from layer to layer, cut both ways along their length.
- A streak the other way round the wrap is not drawn.
- Drawing numbers (`COUNT`, `MARKS`, `AMOUNT`, `FULL`, `SPEED`, `TICKS`): do not move them.
- No kernel argument may be a C++ word.
- Never call `.numpy()` in a frame.
