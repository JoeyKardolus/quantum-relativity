# Measures

What the fit reads off a universe. `definitions/` says what a lump, a region and an
overstretched cell are; `logic/` reads a universe and works the numbers out; `tests/`
checks them against the engine's own books on a real universe.

## Must not get wrong

- **Space is measured inside.** The universe wraps; its size is the sum of its cells'
  lengths, never a distance to an edge.
- **The same cells every time.** The gap between lumps sums the same cells in every
  sample, chosen once by fewest cells, never by length. A route picked for its short,
  full volumes would lengthen by itself as energy evens out.
- **Read once, measure on the host.** `read_universe` is the only function here that
  touches the engine's state; everything else works on its copies and writes nothing.
- **Only the engine's public names** (`physics.engine`) are used, never an axiom's files.
- **An endless cell** (0 quanta) is left out of the size and counted. An endless energy
  volume makes `spent_share` exactly 1.
- **Spent is a share of the volumes' size**, not of the whole size: the joints are far
  longer than the volumes and would hide a tear chain.

## Assumed

- Lump volumes with a way between them are one lump; ways run to all 26 sides, so
  volumes touching at a corner count as joined.
- Where several joints lie between the same two volumes, the gap's chain takes the way
  in the lowest row.
