# physics/scenes/isotropy/logic/

- The lump and the even twin take every step together; the pull's depth is summed as
  twin minus lump after `settle`, so the even grid's own swing cancels step by step.
- Depth is read above the place farthest from the lump (half the lattice away on every
  axis), the same zero `box_field` uses.
- A ratio whose on-axis value is 0 is returned as 0.0, not as a division error: early
  in a run nothing has arrived, and the scene says so with a 0.
- Places must lie under a quarter of the cells a side from the lump, or a place and its
  mirror through the box would be one.
