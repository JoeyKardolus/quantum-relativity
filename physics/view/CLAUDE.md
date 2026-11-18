# physics/view

The window, `python -m physics.view`: one volume as its swirl; `--scene crossing` a red volume, a blue joint, a red volume; `--scene calm` a 4 x 4 x 4 grid that wraps, opened on a flat cut through one layer (L: next layer; C: all layers).

- A cell is drawn as streaks of its own moving fluid, each the way a tracer came. No box, no outline.
- The engine's flow carries the tracers; the viewer corrects nothing. Step 0 is black.
- One painter makes frames and saved pictures (`--pictures`, `--cut`, `--turned`).
- Volumes are balls; the joints fill all the room between them.
- Solid means fast beside the cell's own mean. Where four volumes meet the joints' fluid is slow, so the blue there is faint.
- Measured: 125 frames a second on the calm scene.
