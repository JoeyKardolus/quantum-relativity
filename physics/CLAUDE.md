# physics/

Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(never node, edge, vertex, or merge for joint collapse).

The core is `cells/`: the engine. Energy volumes are tetrahedra (as in causal dynamical
triangulations) in an open cube (not wrapped), shared corners; each inner triangle is the
one joint between its two volumes, an outer triangle borders nothing.
`app/` is the live simulator on it. `network/`, `rules/`, `measure/`, `experiments/`: earlier milestones.

## Must not get wrong

- **Cut once, then only move corners.** `first_mesh.py` (Delaunay of a jittered lattice) is the only
  tessellation. A step moves corners (`relax_corners.py`, least squares: six tetrahedra per corner);
  topology changes only in `collapse_edge.py` and `split_cell.py` (the 1-4 move), both leaving
  tetrahedra with exactly 4 faces. Never re-tessellate.
- **Contract with the app**: `Engine` (`corners`, `tets`, `faces` (f, 3), `face_cells` with -1
  outside, `face_joint`), `step(engine, settings) -> Engine` (pure), `mesh_message`, `frame_arrays`
  (float32 corners, uint8 pull, uint8 push), `engine_stats`. Rules read only the Grid, never a corner.
- **Joints oscillate, undamped**: thickness plus rate (`swing_joints.py`); the push is a hard wall,
  `k R ((R/t)^push_power - 1) / push_power` (`joint_push.py`). No damping. Velocity Verlet
  in substeps from each swing's thinnest turning point; never a step width that changes along a
  swing (that drifts). Rest is `joint_rest` (0.02, about 3% of a volume per joint) x mean length.
- **Total energy never changes**: only the leak moves it, along seams (edges on two or more joints,
  closed past `seal_point`), by pull. Collapse hands energy and quanta on; a split shares by volume.
- **Size follows density** (`rest_volume.py`): `q (q / E)^3`, `q` quanta (the leak never moves them).
- **Honesty.** Never tune a setting, seed or fit window until a result passes; report every number,
  seed, settings and miss. Today (seed 1, 968 volumes, unevenness 0.5, defaults): finite to 300,
  1364 volumes, 137 collapses, 297 splits, size 1.33, 92 lumps.
- **Split by size** (rule 8): past `split_size` (4) x `start_mean_volume`; never by density
  (a split keeps it, so that splits for ever).
- **Page data stays light.** A JSON "mesh" only on start, collapse or split; every
  frame binary (`page_frame.py`: 32-byte header, float32 corners, uint8 pull and push), 6 kB at 968
  tetrahedra (mesh 57 kB), at most 10/s. Drawing: red volume outlines inset 8%; each shared face is one blue
  joint layer filling the gap (red | blue | red); outer faces have no joint, bare red (split with drift).
- **Speed.** A step is about 50 ms at 1000 tetrahedra; `new_engine` 1.5 s at 1000.
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
