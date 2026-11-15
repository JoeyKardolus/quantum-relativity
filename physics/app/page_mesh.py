"""The "mesh" message: what kind each volume is and who touches whom.

Contract: `page_mesh(simulation, mesh_id)` returns {"type": "mesh", "mesh_id", "step",
"count", "kind", "contacts"} for the simulation's current state:
  count     the number of volumes (both kinds); every frame on this mesh carries as many
  kind      per volume 0 energy volume (A), 1 joint (B)
  contacts  flat list [a0, b0, a1, b1, ...]: each touching pair, volume indices < count
Every number is a plain int. Sent only when the topology changes (mesh_key) and on a
reset or a new page.
"""
from __future__ import annotations

import numpy as np

from physics.app.simulation import Simulation


def page_mesh(simulation: Simulation, mesh_id: int) -> dict:
    """The state's kinds and contacts as a JSON-ready message tagged with `mesh_id`."""
    state = simulation.state
    kinds = np.asarray(state.kind, dtype=np.int64).ravel()
    pairs = np.asarray(state.contacts, dtype=np.int64).reshape(-1, 2)
    return {"type": "mesh", "mesh_id": int(mesh_id), "step": simulation.steps, "count": len(kinds),
            "kind": kinds.tolist(), "contacts": pairs.ravel().tolist()}
