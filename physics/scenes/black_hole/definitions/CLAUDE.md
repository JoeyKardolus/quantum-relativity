# physics/scenes/black_hole/definitions/

Formulas on lists; no universe.

- `held_shells` stops at the first boundary energy left through, and never counts the
  last boundary read: past it nothing was read, so reaching it is a limit, not a radius.
- `depth_radius` interpolates between the last shell above the level and the first
  below. When the depth never falls to the level it returns the last shell and the
  scene flags `depth_limit`; such a radius is the box's, not the core's.
- The mark's band, count and span were written before the run. A radius at its limit
  or of 0 is left out of the power; fewer than 4 left is a miss, not a reason to add runs.
