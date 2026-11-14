"""Frame, mesh and settings messages match the contract, and /ws sends all three on connect."""
import json
from dataclasses import fields

from fastapi.testclient import TestClient

from physics.app import Simulation, create_app
from physics.app.frame_message import frame_message
from physics.app.mesh_message import mesh_message
from physics.app.settings_message import settings_message
from physics.rules import RuleSettings

JOINTS = 200
STATS = {"joints", "collapsed", "mean_length_vs_start", "energy_spread", "lumps"}


def test_a_frame_matches_the_contract_shape():
    simulation = Simulation(JOINTS, seed=1)
    simulation.advance()
    frame = frame_message(simulation)
    assert set(frame) == {"type", "step", "positions", "energy", "stats"}
    assert frame["type"] == "frame" and frame["step"] == 1
    assert len(frame["positions"]) == 3 * simulation.network.joint_count
    assert all(round(value, 3) == value for value in frame["positions"])
    assert len(frame["energy"]) == simulation.network.connection_count
    assert all(0.0 <= value <= 1.0 for value in frame["energy"])
    assert set(frame["stats"]) == STATS
    assert frame["stats"]["joints"] == simulation.network.joint_count
    mesh = mesh_message(simulation)
    assert mesh["type"] == "mesh" and len(mesh["pairs"]) == 2 * len(frame["energy"])
    values = settings_message(simulation)["values"]
    assert set(values) == {field.name for field in fields(RuleSettings)} | {"unevenness", "speed"}
    assert all(set(entry) == {"value", "min", "max"} and entry["min"] <= entry["value"] <= entry["max"]
               for entry in values.values())


def test_the_socket_greets_a_page_and_echoes_a_settings_change():
    with TestClient(create_app(JOINTS, seed=1)) as client:
        assert client.get("/").status_code in (200, 503)
        with client.websocket_connect("/ws") as socket:
            kinds = [socket.receive_json()["type"] for _ in range(3)]
            assert kinds == ["settings", "mesh", "frame"]
            socket.send_json({"type": "settings", "values": {"speed": 30}})
            echo = socket.receive_json()
            assert echo["type"] == "settings" and echo["values"]["speed"]["value"] == 30
            socket.send_json({"type": "start"})
            assert socket.receive_json()["type"] == "mesh"
            seen = {socket.receive_json()["type"] for _ in range(3)}
            assert "frame" in seen
            socket.send_json({"type": "pause"})


def test_a_frame_stays_valid_json_once_every_joint_fell_into_one():
    simulation = Simulation(JOINTS, seed=1)
    simulation.apply({"attraction": 1, "energy_flow": 0, "drift": 0})
    for _ in range(50):
        simulation.advance()
        if simulation.network.connection_count == 0:
            break
    assert simulation.network.connection_count == 0
    frame = frame_message(simulation)
    json.dumps(frame, allow_nan=False)
    assert frame["stats"]["mean_length_vs_start"] is None and frame["energy"] == []
