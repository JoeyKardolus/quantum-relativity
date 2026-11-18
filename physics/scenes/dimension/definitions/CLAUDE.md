# physics/scenes/dimension/definitions/

- `reference_grid` builds its graph with `physics.measure`'s `grid_graph`, wrapped and
  with diagonals: the same graph the engine's start lattice of volumes is. The scene's
  test holds the two equal number for number.
- The mark uses `physics.measure`'s own `passes` (within 10% of the known dimension)
  for both estimates; this package adds no second tolerance.
