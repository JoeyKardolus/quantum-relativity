"""The "mesh" message: which corners make which face, and which cells and joint each face has.

Contract: `page_mesh(simulation, mesh_id)` returns {"type": "mesh", "mesh_id", "step",
"corner_count", "cell_count", "joint_count", "faces", "face_cells", "face_joint",
"cell_kind", "start_box"} for the simulation's current engine, from `physics.cells.mesh_message`
and `physics.cells.frame_arrays` (for the cell and joint counts the frames will carry):
  faces       list of corner index lists, one per face, each index < corner_count
  face_cells  per face [a, b]: its two cells, b = -1 for a face on the outside
  face_joint  per face the joint whose joint cell is a side of it (< joint_count), else -1
  cell_kind   per cell (cell_count of them) 0 energy volume, 1 joint cell, 2 seam cell,
              3 corner cell (2 and 3 are joint material where joints meet)
  start_box   [[x, y, z], [x, y, z]]: the corners' bounding box at step 0 (the
              starting cube, drawn faintly so the growth shows)
Every number is a plain int (floats only in start_box). Sent only when the topology
changes (mesh_key) and on a reset or a new page. Raises KeyError naming the missing
key when `physics.cells.mesh_message` lacks one of faces, face_cells, face_joint, cell_kind.
"""
from __future__ import annotations

from physics.app.simulation import Simulation
from physics.cells import frame_arrays, mesh_message

_NEEDED = ("faces", "face_cells", "face_joint", "cell_kind")


def page_mesh(simulation: Simulation, mesh_id: int) -> dict:
    """The engine's mesh topology as a JSON-ready message tagged with `mesh_id`."""
    engine = simulation.engine
    source = mesh_message(engine)
    missing = [name for name in _NEEDED if name not in source]
    if missing:
        raise KeyError(f"physics.cells.mesh_message lacks {missing}")
    corners, pull, push = frame_arrays(engine)
    face_cells = []
    for pair in source["face_cells"]:
        cells = [int(cell) for cell in pair]
        face_cells.append(cells + [-1] * (2 - len(cells)))
    return {"type": "mesh", "mesh_id": int(mesh_id), "step": simulation.steps,
            "corner_count": len(corners), "cell_count": len(pull), "joint_count": len(push),
            "faces": [[int(corner) for corner in face] for face in source["faces"]],
            "face_cells": face_cells,
            "face_joint": [int(joint) for joint in source["face_joint"]],
            "cell_kind": [int(kind) for kind in source["cell_kind"]],
            "start_box": simulation.start_box}
