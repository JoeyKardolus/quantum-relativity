# physics/scenes/volumes/

Reading the energy volumes: their quanta on the lattice they started on, which of them
are lumps, and which are the fullest.

## Must not get wrong

- **Cells only.** Everything is read from `cell_quanta`, `kind` and the ways. No box is read.
- **A lump is a volume holding `LUMP_FACTOR` (2) times an even start volume or more.**
  The factor was written before any run; it is not a setting to tune.
- **The start lattice is only a name.** `volume_field` lays volumes out by the id the
  start gave them. A volume a tear made has no place there and is counted in `left_out`;
  a volume that fell into another reads 0 and its quanta are where they went.
- **A joined region goes through ways**, the engine's own relation between two volumes;
  with 26 joints round a volume, volumes touching by an edge or a corner joint are joined too.
