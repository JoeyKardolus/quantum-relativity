# physics/scenes/gravity/logic/

- `lumps_start` multiplies what each box holds, so every cell keeps its swirl's shape,
  and ends with `physics.engine.lay_start`: the pull at rest for the lumps and every
  joint in balance. Never step a placed universe that was not laid again.
- A kick turns phases by `-kick . place` under a window 2.5 widths wide: fluid flows
  from the phase that is ahead, so the lump moves the way the kick points.
- `lump_path` stops at the step of the first collapse or tear (tears double every step).
- `gravity` builds a fresh universe for each of its five runs; nothing is reused.
