# physics/view/window/

The live window: a pyglet window, its keys, and the thread that steps the source
behind it.

- `definitions/`: the snapshot, the key table, the window's fixed numbers.
- `logic/`: the stepper thread, the title, the GPU driver variables, the window.
- `tests/`: a real window on a plain-array source; keys are pressed through its handler.

## Keys

Starts paused on the slice through the middle, seen straight on. Space starts and
pauses. Drag turns, the wheel zooms. C adds the whole grid (the packed cells as blocks,
seen from the corner) and takes it away. Arrows or [ ] move the slice one layer.
1 / 2 / 3 show energy volumes, joints, both. F undoes the zoom. R resets. Esc quits.

## Must not get wrong

- **GPU under WSL2**: `use_gpu_driver()` must run before pyglet loads, or Mesa draws
  on the CPU.
- **All source work runs on the stepper thread**: steps, layout, stats. The window
  only uploads and draws the newest snapshot.
- **An error stops stepping and shows in the title.** It is printed, never swallowed.
- **Every control republishes while paused**, so a key always shows at once.
- **The title says what the drawing cut off**: "sizes drawn up to 4x", and how many
  packed cells were too spread to lay out as blocks.
- Only `logic/cell_window.py` needs OpenGL. Import it by module path.

## Measured (2026-11-18, synthetic cells on plain arrays, one short run a size)

A calm lattice with three packed lumps, window 1280 x 800. Laying out the slice, then
drawing it: 1.0 million cells 7 ms and 113 frames a second; 4.1 million 14 ms and 187;
16.0 million 37 ms and 127. With the whole grid on (4,000 blocks) a snapshot took 29,
27 and 68 ms. Not measured: the engine at those sizes, whose step and whose copy of
six arrays to the host come on top.
