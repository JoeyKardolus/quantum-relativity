# physics/scenes/placed_start/definitions/

- A lattice place exists only at the start: `place_cell` turns (i, j, k) into the id the
  engine's `start_grid` gave it, `(i * side + j) * side + k`, and refuses a place that is
  not an energy volume (all three even). After the start a cell has only relations.
- `even_fill` keeps quanta whole: the boxes differ by at most one and sum exactly.
- No engine state is read or written here.
