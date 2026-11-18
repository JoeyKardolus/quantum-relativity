# physics/view

The window: one energy volume drawn as its swirl; `--scene crossing` a red volume, a blue joint and a red volume; `--scene grid` four red volumes in a square with four blue joints. Run `PYTHONPATH=. .venv/bin/python -m physics.view`.

- A cell is drawn as streaks of its own moving fluid, each the way a tracer came. No box, no outline.
- A tracer is carried by the engine's own flow. The viewer adds and corrects nothing: step 0 is black.
- Drawn on the card by OpenGL. A frame only sets the eye.
- One painter makes the window's frames and the saved pictures (`--pictures`, `--turned`).
- With several cells the picture is the flat cut through their middles: volumes round, each joint filling what its two leave, out to the corner where joints meet. The distance is the engine's.
- Red and blue follow one rule: solid by the fluid's speed beside its own cell's mean. So a joint's corners, where its fluid stands, are faint.
