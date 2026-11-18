# physics/scenes/placed_start/

The start of the placed scenes (orbit, black hole, two lumps, isotropy): an even grid
with named energy volumes packed. It breaks "nothing placed by hand" on purpose: each
such scene tests one mechanism.

## Must not get wrong

- **Extra energy in a volume of the grid**, never a ball in empty space: there is none.
- **Only quanta are placed.** The volume's size, packing and clock rate follow by the
  engine's own laws; its fluid keeps the phases its swirl was wound with.
- **The pull is not told.** It stays the even grid's resting pull, so the lump's pull
  spreads from step 1 at its finite speed. A scene that reads a pull early reads one
  that has not arrived.
- **The books start here**: `start_books` is read again after packing.

## Assumed

- The engine's public names have no door for placing energy, so `packed_start` writes a
  block with `Grid.set_block` and then calls the engine's own `count_cells`, `size_cells`
  and `pack_cells` from their files. If the engine gains a public placing function, use it.
