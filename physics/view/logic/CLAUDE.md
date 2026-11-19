# View, logic

`tracers.py`, `tracers_across.py` (dots the flow carries), `lay_tracers.py` (`trace`, `move_into`), `line_streaks.py` (their marks), `placed.py` (middles from the joints), `strays.py`, `scene.py`, `swirl_painter.py`, `picture.py`, `cell_window.py` (keys), `window.py` (the door), `save_png.py`, `title_line.py`.

- A tracer moves before each of a step's six passes, or it drifts into the core.
- A tracer's second number names its slot in one float: past some thousand cells it loses its place within a box.
- The joints' distances cannot all be met; `settle` takes the best fit, 40 rounds a frame.
- A growing cluster's cells get tracers evenly, not by their quanta.
- Drawing numbers (`COUNT`, `MARKS`, `AMOUNT`, `FITS`, `ROUNDS`, `WHOLE_UP_TO`, speeds): do not move them.
- No kernel argument is a C++ word.
