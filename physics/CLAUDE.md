# physics/

Quantum-relativity simulation. Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(energy volume, joint, seam, joint collapse, volume split, drift (the leak); never node, edge,
vertex, or merge for joint collapse).

The core is `cells/`: the engine. Energy volumes in an open cube (not wrapped), kept as a mesh of
shared corners; each face is shared by the two volumes of its joint, an outer face borders nothing.
`app/` is the live simulator on it (`python -m physics.app`). `network/`, `rules/`, `measure/` and
`experiments/` are the earlier joint-network milestones.

## Must not get wrong

- **Cut once, then only move corners.** `first_mesh.py` is the only tessellation (pyvoro2's private
  `_core`, at start only; keep `pyvoro2==0.8.0` pinned or re-check it on an upgrade). A step moves
  corners (`relax_corners.py`); topology changes only in `collapse_cells.py` and `split_cell.py`, so
  `neighbour_changes` is 0 by construction. Never re-tessellate in `step`.
- **Contract with the app**: `Engine` (`corners`, `faces`, `face_cells` with -1 outside, `face_joint`,
  `grid`), `step(engine, settings) -> Engine` (pure), `mesh_message`, `frame_arrays` (float32 corners,
  uint8 pull, uint8 push), `engine_stats`. Rules read only the Grid, never a corner.
- **Joints oscillate, undamped**: thickness plus rate (`swing_joints.py`); the push is a hard wall,
  `k R ((R/t)^push_power - 1) / push_power` (`joint_push.py`). No damping. The explicit
  step is stable only while `k (R/t)^(power+1) + squeeze < 4`. Rest is `joint_rest` x the
  volumes' mean length.
- **Total energy never changes**: only the leak moves it, along seams (`leak.py`, `seam_openness.py`,
  closed past `seal_point`), by pull. Collapse keeps both energies; a split shares energy by volume.
  Energy sets the volume a cell relaxes toward (`rest_volume.py`, `(1/E)^3`).
- **Honesty.** Never tune a setting, seed or fit window until a result passes; report every number with
  seed and settings, and every miss. Today (seed 1, 1000 cells, unevenness 0.5, defaults, push power 3):
  4 collapses, 722 splits, size blows up and the pull potential fails at step 151 (miss); splits off, 1
  collapse, size 1.30 at 300. Pull x1.25 runs away to one cell (miss). Energy constant.
- **Page data stays light.** A JSON "mesh" only on start, collapse or split (~250 kB at 1000 cells); every
  frame binary (`page_frame.py`: 32-byte header, float32 corners, uint8 pull and push), about 70 kB at 1000
  cells, at most 10 per second. The drawn gap is a fixed 10% shrink, not the joint's real thickness.
- **Speed.** A step is about 55 ms at 1000 cells, 100 ms at 2000; `new_engine` 4.5 s at 1000.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, each subpackage
  `__init__.py` holds the module map and `__all__`.
- **Port 8765** is the owner's own running app; test on another port (`--port 8766`).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.app             # live page, localhost:8765
PYTHONPATH=. .venv/bin/python -m physics                 # network milestones, minutes
PYTHONPATH=. .venv/bin/python -m physics.viewer          # network gravity scenes as a 3D page
```

`physics/` is exempt from the test register (owner, 2026-11-14; `UNREGISTERED` in
`tools/test_register/_modules.py`): write plain pytest files under `<module>/tests/`, no rows.
