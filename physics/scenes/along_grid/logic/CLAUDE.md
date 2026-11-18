# physics/scenes/along_grid/logic/

Readers of `length` and `pull`; neither writes.

- `middle_distances` adds half the first cell, every whole cell between and half the
  last. A cell holding 0 quanta is endlessly long by the size law: the distance past it
  is `inf`, and the scene reports it as such rather than hiding it.
- `volume_pull` reads the start lattice's ids only (`side ** 3` of them), every second
  one along each axis; a volume that is gone reads whatever the engine left in `pull`.
