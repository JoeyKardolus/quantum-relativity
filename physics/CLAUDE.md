# physics/

Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(never node, edge, vertex, or merge for joint collapse).

The core is `cells/`: the engine. Energy volumes in an open cube (not wrapped), kept as a mesh of
shared corners; each face is shared by the two volumes of its joint, an outer face borders nothing.
`app/` is the live simulator on it. `network/`, `rules/`, `measure/`, `experiments/`: earlier milestones.

## Must not get wrong

- **Cut once, then only move corners.** `first_mesh.py` is the only tessellation (pyvoro2's private
  `_core`, at start; pin `pyvoro2==0.8.0`). A step moves
  corners (`relax_corners.py`); topology changes only in `collapse_cells.py` and `split_cell.py`, so
  `neighbour_changes` is 0 by construction. Never re-tessellate.
- **Contract with the app**: `Engine` (`corners`, `faces`, `face_cells` with -1 outside, `face_joint`,
  `grid`), `step(engine, settings) -> Engine` (pure), `mesh_message`, `frame_arrays` (float32 corners,
  uint8 pull, uint8 push), `engine_stats`. Rules read only the Grid, never a corner.
- **Joints oscillate, undamped**: thickness plus rate (`swing_joints.py`); the push is a hard wall,
  `k R ((R/t)^push_power - 1) / push_power` (`joint_push.py`). No damping. Velocity Verlet
  in substeps counted from each swing's thinnest turning point, so any stiffness stays stable;
  never a step width that changes along a swing (that drifts).
  Rest is `joint_rest` x mean length.
- **Total energy never changes**: only the leak moves it, along seams (`leak.py`, `seam_openness.py`,
  closed past `seal_point`), by pull. Collapse keeps both energies; a split shares energy by volume.
  Energy sets the volume a cell relaxes toward (`rest_volume.py`, `(1/E)^3`).
- **Honesty.** Never tune a setting, seed or fit window until a result passes; report every number,
  seed, settings and miss. Today (seed 1, 1000 cells, unevenness 0.5, defaults): 0 collapses, but
  splits cascade from step ~105 and the pull guard stops the run at ~154 on inverted volumes (miss:
  a half has less energy, so a bigger rest volume, so it splits again; joints stay soft). Splits off:
  size 1.31 at 300. Pull x1.25: one cell (miss). Energy constant.
- **Page data stays light.** A JSON "mesh" only on start, collapse or split (~250 kB at 1000 cells); every
  frame binary (`page_frame.py`: 32-byte header, float32 corners, uint8 pull and push), 70-110 kB at 1000
  cells, at most 10 per second. Drawing: red volume outlines inset 8%; each shared face is one blue
  joint layer filling the gap (red | blue | red); outer faces have no joint, bare red (split with drift).
- **Speed.** A step is about 55 ms at 1000 cells, 100 ms at 2000; `new_engine` 4.5 s at 1000.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, each subpackage
  `__init__.py` holds the module map and `__all__`.
- **Port 8765** is the owner's app; test on `--port 8766`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.app             # live page, localhost:8765
PYTHONPATH=. .venv/bin/python -m physics                 # network milestones, minutes
PYTHONPATH=. .venv/bin/python -m physics.viewer          # network gravity scenes as a 3D page
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<module>/tests/`, no rows.
