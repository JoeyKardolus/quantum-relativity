# View, logic

`paint.py` (the picture, a Warp kernel), `cell_window.py` (window and keys), `window.py` (the door and `--pictures`), `save_png.py`, `title_line.py`.

- `THICKNESS`, `DIM` and `SAMPLES` in `paint.py` are drawing numbers, set by eye once. They change how it looks, never what happens. Do not move them.
- pyglet is imported only when a window opens, so pictures and tests need no display.
- Under WSL2 the two Mesa settings in `window.py` must be set before pyglet loads.
- The title's numbers are read on the host four times a second, not every frame.
