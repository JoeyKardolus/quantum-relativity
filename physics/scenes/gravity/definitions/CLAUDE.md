# physics/scenes/gravity/definitions/

- `scene_lumps` and the constants beside it were written before any run. Change one only
  before a search, with the date and what the old one gave.
- `pass_mark.py` is the phase 1 score: `stands` times the mean of `falls`, `coasts`,
  `fades`, `orbits`. 0 when the universe collapses and 0 when nothing moves.
- `orbit_kick` uses the engine's own fluid law (a phase step `k` moves fluid `2 step_time
  sin k` boxes a step) and `SWIRL_SPEED`, what a standing swirl's fluid keeps of a kick.
- `lump_centre` returns places not wrapped, so a path can be followed past the edge.
