# physics/scenes/along_grid/

What the placed scenes share for measuring along the grid: who lies along a row of the
start lattice, how far apart cells are, the pull at the volumes, and the law beside it.

## Must not get wrong

- **Distance is the sum of the lengths of the cells between**, from middle to middle.
  Never a count of boxes, never a coordinate. In the default start a joint is about 30
  times longer than a volume, so a distance is mostly joints.
- **A pull is read beside its law in the same wrapping box, never bare.** The box's
  copies pull too: `box_depth` sums a point and its copies, `box_field` solves the
  wrapping cube. Even a law with no preferred direction is not the same on and off an
  axis in a cube.
- **Rows are the start lattice's.** A tear or a collapse on a row is not followed; a
  scene counts such events beside its numbers.
- **Only differences of the pull mean anything** (axiom 5's field has no zero).
