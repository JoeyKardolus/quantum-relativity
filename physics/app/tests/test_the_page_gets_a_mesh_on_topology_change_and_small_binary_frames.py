"""The page gets the mesh only when its topology changes and every frame as small binary corner positions."""
import json
import time

import numpy as np
import pytest

import physics.cells as cell_engine

CONTRACT = ("Engine", "EngineSettings", "new_engine", "step", "mesh_message", "frame_arrays", "engine_stats")
if not all(hasattr(cell_engine, name) for name in CONTRACT):
    pytest.skip("physics.cells does not have the mesh contract yet (Engine with corners, faces, face_cells, "
                "face_joint; step, mesh_message, frame_arrays, engine_stats); the integrate step runs these "
                "for real", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from physics.app import Simulation, create_app  # noqa: E402
from physics.app.mesh_key import mesh_key  # noqa: E402
from physics.app.page_frame import FRAME_MAGIC, HEADER_WORDS, page_frame  # noqa: E402
from physics.app.page_mesh import page_mesh  # noqa: E402
from physics.app.runner import FRAMES_PER_SECOND  # noqa: E402
from physics.app.settings_message import settings_message  # noqa: E402
from physics.app.simulation import RULE_NAMES  # noqa: E402
from physics.app.stats_message import stats_message  # noqa: E402

CELLS = 80
STATS = {"cells", "joints", "collapses", "splits", "size_vs_start", "total_energy", "lumps", "sealed_lumps",
         "neighbour_changes"}
MEGABYTE = 1 << 20


def _read_frame(data: bytes) -> dict:
    header = np.frombuffer(data, dtype="<u4", count=HEADER_WORDS)
    magic, step, mesh_id, corner_count, cell_count, joint_count = (int(word) for word in header[:6])
    start = HEADER_WORDS * 4
    corners = np.frombuffer(data, dtype="<f4", count=corner_count * 3, offset=start).reshape(-1, 3)
    start += corner_count * 12
    pull = np.frombuffer(data, dtype=np.uint8, count=cell_count, offset=start)
    push = np.frombuffer(data, dtype=np.uint8, count=joint_count, offset=start + cell_count)
    assert len(data) == start + cell_count + joint_count
    return {"magic": magic, "step": step, "mesh_id": mesh_id, "corners": corners, "pull": pull, "push": push}


def test_the_mesh_message_indexes_real_corners_and_names_both_cells_and_the_joint_of_each_face():
    simulation = Simulation(CELLS, seed=2)
    message = page_mesh(simulation, mesh_id=7)
    assert message["type"] == "mesh" and message["mesh_id"] == 7 and message["step"] == 0
    corners, pull, push = cell_engine.frame_arrays(simulation.engine)
    assert message["corner_count"] == len(corners) == len(simulation.engine.corners)
    assert message["cell_count"] == len(pull) and message["joint_count"] == len(push)
    faces, face_cells, face_joint = message["faces"], message["face_cells"], message["face_joint"]
    assert len(faces) == len(face_cells) == len(face_joint) > 0
    for face, cells, joint in zip(faces, face_cells, face_joint):
        assert len(face) >= 3 and all(0 <= corner < message["corner_count"] for corner in face)
        assert len(cells) == 2 and 0 <= cells[0] < message["cell_count"] and -1 <= cells[1] < message["cell_count"]
        if cells[1] == -1:
            assert joint == -1
        else:
            assert cells[0] != cells[1] and 0 <= joint < message["joint_count"]
    assert {joint for joint in face_joint if joint >= 0} == set(range(message["joint_count"]))
    assert {cell for cells in face_cells for cell in cells if cell >= 0} == set(range(message["cell_count"]))
    low, high = (np.array(corner) for corner in message["start_box"])
    assert np.all(low < high)
    json.dumps(message, allow_nan=False)


def test_the_frame_is_the_engines_corners_pull_and_push_as_binary():
    simulation = Simulation(CELLS, seed=2)
    simulation.advance()
    frame = _read_frame(page_frame(simulation, mesh_id=3))
    assert frame["magic"] == FRAME_MAGIC and frame["step"] == 1 and frame["mesh_id"] == 3
    corners, pull, push = cell_engine.frame_arrays(simulation.engine)
    assert np.array_equal(frame["corners"], np.asarray(corners, dtype=np.float32).reshape(-1, 3))
    assert np.all(np.isfinite(frame["corners"]))
    assert np.array_equal(frame["pull"], pull) and np.array_equal(frame["push"], push)


def test_a_frame_at_2000_cells_is_well_under_a_megabyte():
    simulation = Simulation(2000, seed=1)
    size = len(page_frame(simulation, mesh_id=1))
    assert size < MEGABYTE / 2, f"a frame is {size} bytes at 2000 cells"


def test_the_mesh_key_changes_only_with_a_collapse_or_split():
    simulation = Simulation(CELLS, seed=3)
    assert mesh_key(simulation.engine) == mesh_key(simulation.engine)
    for _ in range(3):
        before_key, before = mesh_key(simulation.engine), cell_engine.engine_stats(simulation.engine)
        simulation.advance()
        after = cell_engine.engine_stats(simulation.engine)
        events = (after["collapses"] - before["collapses"]) + (after["splits"] - before["splits"])
        assert (mesh_key(simulation.engine) != before_key) == (events > 0)


def test_a_reset_is_reproducible_by_seed_and_settings_are_clamped():
    first, second = Simulation(CELLS, seed=4), Simulation(CELLS, seed=4)
    for _ in range(2):
        first.advance()
        second.advance()
    assert first.steps == 2
    assert page_frame(first, 1) == page_frame(second, 1) and page_mesh(first, 1) == page_mesh(second, 1)
    first.reset(5, CELLS)
    assert first.steps == 0
    first.apply({"push_limit": 5.0})
    assert first.settings.push_limit == first.ranges["push_limit"][1] < 1.0
    with pytest.raises(ValueError):
        first.apply({"no_such_rule": 1.0})


def test_the_settings_message_names_every_rule_then_unevenness_and_speed():
    values = settings_message(Simulation(CELLS, seed=1))["values"]
    assert list(values) == [*RULE_NAMES, "unevenness", "speed"]
    for spec in values.values():
        assert spec["min"] <= spec["value"] <= spec["max"]


def test_the_stats_message_carries_the_listed_stats_and_the_run_state():
    message = stats_message(Simulation(CELLS, seed=1), running=True)
    assert message["type"] == "stats" and message["running"] is True and message["step"] == 0
    assert STATS <= set(message["stats"])
    json.dumps(message, allow_nan=False)


def test_the_socket_greets_then_runs_at_most_ten_frames_a_second_and_answers_a_reset():
    with TestClient(create_app(CELLS, seed=1)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "settings"
            mesh = socket.receive_json()
            assert mesh["type"] == "mesh"
            frame = _read_frame(socket.receive_bytes())
            assert frame["mesh_id"] == mesh["mesh_id"] and frame["step"] == 0
            assert len(frame["corners"]) == mesh["corner_count"]
            assert socket.receive_json()["running"] is False
            socket.send_json({"type": "settings", "values": {"speed": 60.0}})
            assert socket.receive_json()["values"]["speed"]["value"] == 60.0
            socket.send_json({"type": "start"})
            assert socket.receive_json()["running"] is True
            frame_times, steps = [], []
            while len(frame_times) < 4:
                message = socket.receive()
                if message.get("bytes") is not None:
                    frame_times.append(time.monotonic())
                    steps.append(_read_frame(message["bytes"])["step"])
            assert steps == sorted(steps) and steps[-1] > steps[0]
            assert frame_times[-1] - frame_times[0] >= 3 / FRAMES_PER_SECOND * 0.9
            socket.send_json({"type": "pause"})
            socket.send_json({"type": "reset", "seed": 3, "cells": CELLS})
            kinds, reset_mesh = [], None
            while reset_mesh is None or kinds[-1] != "stats":
                message = socket.receive()
                if message.get("bytes") is not None:
                    kinds.append("frame")
                    continue
                parsed = json.loads(message["text"])
                kinds.append(parsed["type"])
                if parsed["type"] == "mesh" and parsed["step"] == 0:
                    reset_mesh = parsed
            assert reset_mesh["mesh_id"] > mesh["mesh_id"]
            assert kinds[-4:] == ["settings", "mesh", "frame", "stats"]


def test_the_page_reads_mesh_and_binary_frames_and_has_no_shapes_code():
    with TestClient(create_app(CELLS, seed=1)) as client:
        text = client.get("/").text
    assert "three.js/r128/three.min.js" in text
    assert 'message.type === "mesh"' in text and 'binaryType = "arraybuffer"' in text
    assert "0x31465251" in text and f"{FRAME_MAGIC:#x}" == "0x31465251"
    assert "#offline[hidden] { display: none; }" in text
    for gone in ('"shapes"', "tracked", "SHRINK"):
        assert gone not in text
