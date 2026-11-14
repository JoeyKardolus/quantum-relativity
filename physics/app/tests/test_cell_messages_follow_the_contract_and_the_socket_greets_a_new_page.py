"""Frame, mesh and settings messages match the cell contract, and /ws sends all three on connect."""
import json
from dataclasses import fields

import pytest

pytest.importorskip("physics.cells", reason="physics.cells (the cell grid) is not built yet")

from fastapi.testclient import TestClient  # noqa: E402

from physics.app import Simulation, create_app  # noqa: E402
from physics.app.frame_message import frame_message  # noqa: E402
from physics.app.mesh_message import mesh_message  # noqa: E402
from physics.app.settings_message import settings_message  # noqa: E402
from physics.cells import CellSettings  # noqa: E402

CELLS = 200
STATS = {"cells", "collapsed", "size_vs_start", "energy_spread", "lumps", "collapses"}


def test_a_frame_matches_the_contract_shape():
    simulation = Simulation(CELLS, seed=1)
    simulation.advance()
    frame = frame_message(simulation)
    grid = simulation.grid
    assert set(frame) == {"type", "step", "positions", "radius", "energy", "squeeze", "stats"}
    assert frame["type"] == "frame" and frame["step"] == 1
    assert len(frame["positions"]) == 3 * grid.cell_count
    assert all(round(value, 3) == value for value in frame["positions"])
    assert len(frame["radius"]) == len(frame["energy"]) == grid.cell_count
    assert all(value > 0 for value in frame["radius"])
    assert len(frame["squeeze"]) == len(grid.joints)
    assert all(0.0 <= value <= 1.0 for value in frame["energy"] + frame["squeeze"])
    assert set(frame["stats"]) == STATS
    assert frame["stats"]["collapses"] == simulation.start_cells - grid.cell_count
    json.dumps(frame, allow_nan=False)
    mesh = mesh_message(simulation)
    assert set(mesh) == {"type", "cells", "pairs"} and mesh["cells"] == grid.cell_count
    assert len(mesh["pairs"]) == 2 * len(frame["squeeze"])
    values = settings_message(simulation)["values"]
    assert set(values) == {field.name for field in fields(CellSettings)} | {"unevenness", "speed"}
    assert all(set(entry) == {"value", "min", "max"} and entry["min"] <= entry["value"] <= entry["max"]
               for entry in values.values())


def test_the_socket_greets_a_page_and_echoes_a_settings_change():
    with TestClient(create_app(CELLS, seed=1)) as client:
        assert client.get("/").status_code in (200, 503)
        with client.websocket_connect("/ws") as socket:
            kinds = [socket.receive_json()["type"] for _ in range(4)]
            assert kinds == ["settings", "mesh", "frame", "slice"]
            socket.send_json({"type": "settings", "values": {"speed": 30}})
            echo = socket.receive_json()
            assert echo["type"] == "settings" and echo["values"]["speed"]["value"] == 30
            socket.send_json({"type": "start"})
            assert socket.receive_json()["type"] == "mesh"
            seen = {socket.receive_json()["type"] for _ in range(3)}
            assert "frame" in seen
            socket.send_json({"type": "pause"})
            socket.send_json({"type": "reset", "seed": 2, "cells": CELLS})
            kinds = []
            while kinds[-4:] != ["settings", "mesh", "frame", "slice"]:
                kinds.append(socket.receive_json()["type"])
