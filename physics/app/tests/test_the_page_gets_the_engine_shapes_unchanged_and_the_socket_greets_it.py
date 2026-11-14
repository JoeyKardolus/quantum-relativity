"""The app sends the engine's own shapes unchanged, in the message contract, and greets a new page."""
import json
import math

import pytest

import physics.cells as cell_engine

CONTRACT = ("Engine", "EngineSettings", "new_engine", "shapes", "engine_stats")
if not all(hasattr(cell_engine, name) for name in CONTRACT):
    pytest.skip("physics.cells does not have the engine contract yet (Engine, EngineSettings, new_engine, "
                "shapes, engine_stats); the integrate step runs these for real", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from physics.app import Simulation, create_app  # noqa: E402
from physics.app.settings_message import settings_message  # noqa: E402
from physics.app.shapes_message import shapes_message  # noqa: E402
from physics.app.simulation import RULE_NAMES  # noqa: E402

CELLS = 80
STATS = {"cells", "joints", "collapses", "size_vs_start", "energy_spread", "lumps", "sealed_lumps"}


def _good_polygon(points) -> bool:
    return (isinstance(points, list) and len(points) >= 9 and len(points) % 3 == 0
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in points))


def test_the_shapes_message_is_the_engines_shapes_with_its_stats():
    simulation = Simulation(CELLS, seed=2)
    simulation.advance()
    message = shapes_message(simulation)
    assert set(message) == {"type", "step", "cells", "joints", "stats"}
    assert message["type"] == "shapes" and message["step"] == 1
    drawn = cell_engine.shapes(simulation.engine)
    assert message["cells"] == drawn["cells"] and message["joints"] == drawn["joints"]
    assert STATS <= set(message["stats"])
    ids = [cell["id"] for cell in message["cells"]]
    assert len(ids) == len(set(ids)) == message["stats"]["cells"]
    for cell in message["cells"]:
        assert set(cell) >= {"id", "faces", "pull"} and 0.0 <= cell["pull"] <= 1.0
        assert cell["faces"] and all(_good_polygon(face) for face in cell["faces"])
    assert len(message["joints"]) == message["stats"]["joints"]
    for joint in message["joints"]:
        assert set(joint) >= {"a", "b", "polygon", "push", "thickness", "sealed"}
        assert isinstance(joint["sealed"], bool)
        assert joint["a"] in ids and joint["b"] in ids and joint["a"] != joint["b"]
        assert _good_polygon(joint["polygon"]) and 0.0 <= joint["push"] <= 1.0
        assert math.isfinite(joint["thickness"]) and joint["thickness"] >= 0.0
    json.dumps(message, allow_nan=False)


def test_a_reset_is_reproducible_by_seed_and_settings_are_clamped():
    first, second = Simulation(CELLS, seed=4), Simulation(CELLS, seed=4)
    for _ in range(2):
        first.advance()
        second.advance()
    assert first.steps == 2
    assert shapes_message(first) == shapes_message(second)
    first.reset(5, CELLS)
    assert first.steps == 0
    first.apply({"push_limit": 5.0})
    assert first.settings.push_limit == first.ranges["push_limit"][1] < 1.0
    with pytest.raises(ValueError):
        first.apply({"no_such_rule": 1.0})


def test_the_settings_message_names_every_rule_then_unevenness_and_speed():
    values = settings_message(Simulation(CELLS, seed=1))["values"]
    assert list(values) == [*RULE_NAMES, "unevenness", "speed"]
    assert {"attraction", "joint_push", "push_limit", "joint_rest", "leak"} <= set(RULE_NAMES)
    for spec in values.values():
        assert spec["min"] <= spec["value"] <= spec["max"]


def test_the_socket_greets_with_settings_and_shapes_and_answers_a_reset():
    with TestClient(create_app(CELLS, seed=1)) as client:
        with client.websocket_connect("/ws") as socket:
            assert [socket.receive_json()["type"] for _ in range(2)] == ["settings", "shapes"]
            socket.send_json({"type": "reset", "seed": 3, "cells": CELLS})
            settings, shapes = socket.receive_json(), socket.receive_json()
            assert settings["type"] == "settings" and shapes["type"] == "shapes"
            assert shapes["step"] == 0 and shapes["stats"]["cells"] == len(shapes["cells"])
            socket.send_json({"type": "settings", "values": {"leak": 0.0}})
            assert socket.receive_json()["values"]["leak"]["value"] == 0.0


def test_the_page_draws_from_shapes_and_has_no_geometry_of_its_own():
    page = TestClient(create_app(CELLS, seed=1))
    with page as client:
        text = client.get("/").text
    assert "three.js/r128/three.min.js" in text and 'message.type === "shapes"' in text
    assert "#offline[hidden] { display: none; }" in text
    for gone in ('"frame"', '"walls"', '"mesh"', "positions"):
        assert gone not in text
