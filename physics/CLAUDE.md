# physics/

Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(never node, edge, vertex, or merge for joint collapse).

The core is `cells/`: the engine, in an open cube (not wrapped). A tetrahedral skeleton says who
neighbours whom; from it `joint_cells.py` cuts the cells: each energy volume a tetrahedron with its
own corners, each joint a cell on the face between two, seam and corner cells of joint material
where joints meet.
`app/` is the live simulator on it. `network/`, `rules/`, `measure/`, `experiments/`: earlier milestones.

## Must not get wrong

- **Cut once, then only move corners.** `first_mesh.py` (Delaunay) is the only tessellation. A step
  moves the skeleton (`relax_skeleton.py`), the corners ride on it, then settle (`relax_corners.py`,
  sizes only, no move past a quarter of the smallest cell). Topology changes only in `collapse_edge.py`
  and `split_cell.py` on the skeleton; the cells are then cut again, untouched corners kept.
- **Contract with the app**: `Engine` (`corners`, `tets`, `faces` (f, 3), `face_cells` with -1
  outside, `face_joint`, `cell_kind`), `step` (pure), `mesh_message`, `frame_arrays` (float32
  corners, uint8 pull per cell, uint8 push per joint), `engine_stats`. Rules read only the Grid.
- **Joints oscillate, undamped**: thickness plus rate (`swing_joints.py`); the push is a hard wall,
  `k R ((R/t)^push_power - 1) / push_power` (`joint_push.py`). No damping. Velocity Verlet
  in substeps from each swing's thinnest turning point; never a step width that changes along a
  swing (that drifts). Rest is `joint_rest` (0.02, about 3% of a volume per joint) x mean length.
- **Total energy never changes**: only the leak moves it, along seams (seam cells two or more joint
  cells touch, closed past `seal_point`), by pull. Collapse hands energy and quanta on; a split shares by volume.
- **Size follows density** (`rest_volume.py`): `q (q / E)^3`, `q` quanta (the leak never moves them).
- **Honesty.** Never tune a setting, seed or fit window until a result passes; report every number,
  seed, settings and miss. Today (seed 1, 968 volumes, unevenness 0.5, defaults): finite to 300,
  894 volumes, 117 collapses, 126 splits, size 1.01, 37 lumps, 1 sealed.
- **Split by size** (rule 8): past `split_size` (4) x `start_mean_volume`; never by density
  (a split keeps it: endless splits).
- **Page data.** A JSON "mesh" only on start, collapse or split (1.7 MB at 968); every frame
  binary (`page_frame.py`), 161 kB, at most 10/s. Drawing from `cell_kind`, every cell
  at its real shape (no inset): red volumes, blue joint cells, dim blue seam and corner cells.
- **Speed.** The app 1.75 steps/s at 1000 energy volumes; `new_engine` 1.4 s.
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
