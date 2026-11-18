# physics/scenes/spin/logic/

- `spin` reads each watched cell's `quanta` and `phase` blocks after every step and
  hands them to `physics.measure`'s `cell_turns`, across the cell's own `swirl_axis`.
- A cell that has collapsed or torn away holds 0 in every box and reads 0 turns: it
  shows as not held, which is the truth about it.
- Reading two blocks per watched cell per step is a copy from the GPU each; keep `count`
  small on long runs.
