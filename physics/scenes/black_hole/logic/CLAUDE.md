# physics/scenes/black_hole/logic/

- `core_run` is used for the twin and for every core alike, so their readings are taken
  at the same steps and on the same cells. Do not give the twin a shorter run.
- Rows and shells are the start lattice's. A tear or collapse near the core is counted
  (`collapses`, `merges`, `tears`), not followed: after one, the row's ids may be gone
  and a distance may read `inf`.
- The pull is read every `every` steps over the window because it is a wave that keeps
  swinging; one reading at the end is a phase of the swing, not a depth.
- One run per energy, each a full run: the scene's cost is its energies plus one.
