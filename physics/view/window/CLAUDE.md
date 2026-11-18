# physics/view/window/

The live window: a pyglet window, its keys, and the thread that steps the source
behind it.

- `definitions/`: the snapshot, the key table, the window's fixed numbers.
- `logic/`: the stepper thread, the title, the GPU driver variables, the window.
- `tests/`: a real window on a plain-array source; keys are pressed through its handler.

## Keys

Starts paused on the whole grid. Space starts and pauses. Drag turns, the wheel zooms.
C switches between the whole grid and the slice. Arrows or [ ] move the slice one
layer. 1 / 2 / 3 show energy volumes, joints, both. F fits the view. R resets. Esc quits.

## Must not get wrong

- **GPU under WSL2**: `use_gpu_driver()` must run before pyglet loads, or Mesa draws
  on the CPU.
- **All source work runs on the stepper thread**: steps, layout, stats. The window
  only uploads and draws the newest snapshot.
- **An error stops stepping and shows in the title.** It is printed, never swallowed.
- **Every control republishes while paused**, so a key always shows at once.
- Only `logic/cell_window.py` needs OpenGL. Import it by module path.

## Not done

- No frame rates were measured on the rebuilt engine. After many collapses one layout
  takes seconds at 32 cells a side (`layout/CLAUDE.md`), so the window stalls there.
