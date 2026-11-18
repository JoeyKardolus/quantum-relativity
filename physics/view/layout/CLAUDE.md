# physics/view/layout/

Lays the cells out from their sizes. A cell has no place, only neighbours and a
length, so this directory holds the one rule for where each cell is drawn.

- `definitions/`: what a slot, a width, a drawn length, the perfect grid, the pull and
  a piece are.
- `logic/`: the typical lengths, every cell's slot, the warped corners, the blocks.
- `tests/`: the layout as a whole on plain arrays: no overlap, no gap, pattern order
  kept, a lump moves only what is within its reach.

## How the grid is drawn

The perfect A B A B grid of the typical (median) volume and joint, plus a shift: each
slot's difference from typical moves the corner planes of its own row by fixed weights
that fade to nothing at `REACH` places. A weighted sum: no fit, no memory of the frame
before. `logic/cell_layout.py` has the reasons and the proofs.

## Must not get wrong

- **A cell never changes slot.** Every place of the start lattice is a slot. A start
  cell stays in its own; the joint and volume a tear makes are stripes inside the torn
  volume's slot; a gone cell leaves its slot empty, drawn at a quarter width.
- **Out of reach, nothing moves.** A corner further than `REACH` from every changed
  cell is bit for bit where it was. Never add a global step (a fit, a rescale to a
  frame's extremes): that is what made the picture jump.
- **The frame is the perfect grid's.** It changes only with the typical lengths, evenly.
- **Never fold.** Every edge keeps at least 1 / (2 `LIMIT`) of its regular length by
  construction. Bends between rows are checked every layout and the warp is eased when
  one would turn a block (or a slice face) inside out; `limited` then says so.
- **One layer costs one layer.** `cell_pieces(layer=k)` reads five layers, whatever the
  lattice's size. The whole grid lays out only the box round the wanted cells, and
  nothing past `LAID_OUT_MOST` places.
- No OpenGL and no engine is imported here; tests run on plain arrays.

## Assumed

- A joint is as long as the volumes it lies beside along the axes where it is not
  itself (a cell has a length, not a shape; the pattern gives the shape).
- A tear's stripes lie along the torn volume's swirl axis.

## What the drawing does not show

- **Size is shown against the neighbourhood, not exactly.** A lone small cell shows
  part of its size, a region wider than `REACH` is drawn regular inside, and a cell
  larger than typical is drawn hardly larger. Sizes beyond `LIMIT` times typical are
  cut off; the title says "sizes drawn up to 4x". The colour carries packing exactly.
- Real runs are checked in `source/tests/`: no fold, no overlap, nothing jumps.
