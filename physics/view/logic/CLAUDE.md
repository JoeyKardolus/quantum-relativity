# View, logic

`tracers.py`, `tracers_across.py`, `lay_tracers.py`, `line_streaks.py` (marks, the phase's crest, one ring alone), `placed.py`, `fit.py`, `strays.py`, `scene.py`, `cell_keys.py`, `swirl_painter.py`, `picture.py`, `cell_window.py`, `window.py`, `save_png.py`, `title_line.py`.

- A tracer's place is (layer, around the core, along the ring); it follows the engine's own pairs, pass by pass, with that pass's turn.
- A tracer's second number names its slot in one float: past thousands of cells it loses it.
- The crest is the fluid's phase, never painted motion. Marks under `FAINT` are drawn by share.
- `HELD` keeps the grid a grid; the title says how far the drawing is off.
- Drawing numbers (`COUNT`, `MARKS`, `AMOUNT`, `CREST`, `FAINT`, `FITS`, `ROUNDS`, `HELD`, `WHOLE_UP_TO`): do not move them.
