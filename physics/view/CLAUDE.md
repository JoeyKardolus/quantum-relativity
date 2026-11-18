# physics/view

The window: one energy volume drawn as its swirl, or with `--scene crossing` a red volume, a blue joint and a red volume. Run `PYTHONPATH=. .venv/bin/python -m physics.view`.

- The red is the swirl itself, not a volume: streaks, each the way a tracer came. A streak is solid by the energy moving there (speed squared) and clear where the fluid stands. No box, no cube face, no outline: a cell has no shape of its own.
- A tracer is carried by the engine's own flow. The viewer adds nothing and corrects nothing: still fluid draws nothing, so step 0 is black and the window opens running.
- Drawn on the card by OpenGL. A tick sends the streaks' places once; no picture comes back. A frame only sets the eye, so dragging stays smooth whatever the engine does.
- The eye never goes into a kernel. The old painter marched lines through the boxes and a line that missed the cell read far outside the array (the CUDA error 700).
- One painter makes the window's frames and the saved pictures (`--pictures`, which needs a screen to make its hidden window).
- G sends the one cell to calm and back. The picture must not jump.
- In the line each cell is drawn at its size by the size law, touching its neighbours. A straight red line through the joint is fluid that crossed in one step.
