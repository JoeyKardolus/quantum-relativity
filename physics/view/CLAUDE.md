# physics/view

The window, `python -m physics.view`: one volume as its ball whirl; `--scene crossing` a volume, a joint, a volume; `--scene grow` one volume whose leak tears off layer after layer (X, Y, Z; C: whole or flat cut); `--scene lumps` two fuller regions.

- A cell is drawn as its fluid's tracers, carried by the engine's flow: a streak the way one came, a dot where it stands. No box.
- Cells are drawn at their sizes, where their joints put them.
- Above 500 cells the flat cut is drawn; a tracer that leaves it is laid in it again.
- Measured: 70 to 80 frames a second up to 2,000 cells, 23 at 9,500.
