# physics/view/layout/

Lays the cells out from their sizes. A cell has no place, only neighbours and a
length, so this directory holds the one rule for where each cell is drawn.

- `definitions/`: what a slot, a width, a drawn length and a piece are.
- `logic/`: gives every cell its slot, fits the grid's corners, cuts the blocks.
- `tests/`: the layout as a whole on plain arrays: no overlap, no gap, pattern order kept.

## Must not get wrong

- **A cell never changes slot.** Every place of the start lattice is a slot. A start
  cell stays in its own; the joint and volume a tear makes are stripes inside the torn
  volume's slot; a cell that is gone leaves its slot empty and the grid closes up.
- **Sizes warp the grid, nothing else.** Slots share corners, so there is never a gap.
  The corners are a best fit, so a drawn size is close to the cell's length, not equal.
- **Boxes are never laid out.** Only cells.
- No OpenGL and no engine is imported here; tests run on plain arrays.

## Assumed

- A joint is as long as the volumes it lies beside along the axes where it is not
  itself (a cell has a length, not a shape; the pattern gives the shape).
- A living cell with no energy is endlessly long by the size law. It is drawn as long
  as the longest cell that has a length.
- A tear's stripes lie along the torn volume's swirl axis, which is the axis the
  engine tears along.

## Measured (2026-11-18, seed 1, real engine runs)

"Folded" is a cell with an edge pointing backwards along its own axis. "Miss" is how
far an energy volume's drawn width is from its length.

- Start, 16 and 32 a side: none folded; miss 2.5% on average, 13% at most.
- Held run, 16 a side, step 24: none folded; miss 15% on average, 73% at most.
- Uneven run, 16 a side: step 4 (109 tears) none of 4,314; step 8 (2,730 tears) 20 of
  9,556.

## Not done

- **Cells fold after mass collapses.** Default run, step 4: 142 of 1,668 cells at 16 a
  side (1,924 collapses), 469 of 11,836 at 32 a side (16,860 collapses).
- **The fit is slow there too**: 32 a side, default run, step 4: 13 to 18 s for one
  layout (0.1 s at the start), because the quick solve gives up and the direct one runs.
