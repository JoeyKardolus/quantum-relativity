# physics/scenes/expansion/logic/

`expansion` builds the natural start, runs it with every rule on and reads the size.

- The start is the settings' own (`start` replaces start amounts); nothing is placed.
- It needs at least one step: a rate needs two samples.
- `books_gap` is the largest miss of the engine's `book_law`; 0 is exact. The total is
  not fixed, so "energy conserved" is the wrong check: the books are.
- `hubble_gap` runs this scene and reads its `size_rate`; keep that key.
