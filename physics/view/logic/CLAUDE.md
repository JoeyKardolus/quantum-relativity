# View, logic

`tracers.py`, `tracers_across.py` (dots the flow carries), `line_streaks.py` (their marks), `placed.py` (middles from relations), `scene.py` (scenes, `Seen`), `swirl_painter.py`, `picture.py`, `cell_window.py` (keys), `window.py` (the door), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes, or it drifts into the core.
- A tracer's second number names its slot, `across` a slot, in one float: keep it 0.001 off the slot's far edge.
- The joints' distances cannot all be met at once; `settle` takes the middles that fit best.
- A streak from ball to ball is a tracer that crossed.
- Drawing numbers (`COUNT`, `MARKS`, `AMOUNT`, `FITS`, `ROUNDS`, speeds): do not move them.
- No kernel argument may be a C++ word.
