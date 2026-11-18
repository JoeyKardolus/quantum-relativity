# physics/view

The window: one energy volume drawn as its swirl, or with `--scene crossing` a red volume, a blue joint and a red volume. Run `PYTHONPATH=. .venv/bin/python -m physics.view`.

- A cell is drawn as streaks of its own moving fluid, each the way a tracer came. No box, no cube face, no outline: a cell has no shape of its own.
- A tracer is carried by the engine's own flow. The viewer adds and corrects nothing: step 0 is black.
- Drawn on the card by OpenGL. A frame only sets the eye; the eye never goes into a kernel.
- One painter makes the window's frames and the saved pictures (`--pictures`).
- In the line each cell is drawn at its size, touching its neighbours, so a shrinking joint draws the volumes closer.
- Red and blue follow one rule: solid by the fluid's speed beside its own cell's mean.
