"""The page gets kinds and contacts only when the topology changes and every frame as small binary positions, sizes and colours."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import physics.abab as abab
except ImportError:
    abab = None
CONTRACT = ("State", "Settings", "new_state", "step", "stats")
if abab is None or not all(hasattr(abab, name) for name in CONTRACT):
    pytest.skip("physics.abab does not have the engine contract yet (State, Settings, new_state, step, stats); "
                "the app is coded against it and these run once it lands", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from physics.app import Simulation, create_app  # noqa: E402
from physics.app.carry_positions import carry_positions  # noqa: E402
from physics.app.colour_ranks import colour_ranks  # noqa: E402
from physics.app.mesh_key import mesh_key  # noqa: E402
from physics.app.page_frame import FRAME_MAGIC, HEADER_WORDS, page_frame  # noqa: E402
from physics.app.page_mesh import page_mesh  # noqa: E402
from physics.app.runner import FRAMES_PER_SECOND  # noqa: E402
from physics.app.settings_message import settings_message  # noqa: E402
from physics.app.settle_positions import settle_positions  # noqa: E402
from physics.app.stats_message import stats_message  # noqa: E402
from physics.app.volume_alignment import volume_alignment  # noqa: E402

VOLUMES = 160
MEGABYTE = 1 << 20


def _read_frame(data: bytes) -> dict:
    header = np.frombuffer(data, dtype="<u4", count=HEADER_WORDS)
    magic, step, mesh_id, count = (int(word) for word in header[:4])
    start = HEADER_WORDS * 4
    positions = np.frombuffer(data, dtype="<f4", count=count * 3, offset=start).reshape(-1, 3)
    sizes = np.frombuffer(data, dtype="<f4", count=count, offset=start + count * 12)
    colours = np.frombuffer(data, dtype=np.uint8, count=count, offset=start + count * 16)
    assert len(data) == start + count * 17
    return {"magic": magic, "step": step, "mesh_id": mesh_id, "positions": positions, "sizes": sizes,
            "colours": colours}


def test_the_mesh_message_is_the_kinds_and_the_contacts_of_the_state():
    simulation = Simulation(VOLUMES, seed=2)
    message = page_mesh(simulation, mesh_id=7)
    assert message["type"] == "mesh" and message["mesh_id"] == 7 and message["step"] == 0
    assert message["count"] == len(simulation.state.kind) == len(message["kind"])
    assert message["kind"] == np.asarray(simulation.state.kind).tolist()
    assert message["contacts"] == np.asarray(simulation.state.contacts).ravel().tolist()
    pairs = np.array(message["contacts"]).reshape(-1, 2)
    kinds = np.array(message["kind"])
    assert len(pairs) > 0 and np.all(kinds[pairs[:, 0]] != kinds[pairs[:, 1]])
    json.dumps(message, allow_nan=False)


def test_the_frame_is_display_positions_sizes_and_rank_colours_as_binary():
    simulation = Simulation(VOLUMES, seed=2)
    simulation.advance()
    frame = _read_frame(page_frame(simulation, mesh_id=3))
    assert frame["magic"] == FRAME_MAGIC and frame["step"] == 1 and frame["mesh_id"] == 3
    assert np.array_equal(frame["positions"], simulation.positions.astype(np.float32))
    assert np.all(np.isfinite(frame["positions"]))
    assert np.array_equal(frame["sizes"], np.asarray(simulation.state.size, dtype=np.float32))
    assert np.array_equal(frame["colours"], colour_ranks(simulation.state))
    kinds = np.asarray(simulation.state.kind)
    for kind in (0, 1):
        ranked = frame["colours"][kinds == kind]
        assert ranked.min() == 0 and ranked.max() == 255


def test_a_frame_at_4000_volumes_is_well_under_a_megabyte():
    size = len(page_frame(Simulation(4000, seed=1), mesh_id=1))
    assert size < MEGABYTE / 4, f"a frame is {size} bytes at 4000 volumes"


def test_the_layout_puts_touching_volumes_at_the_sum_of_their_radii():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    contacts = np.array([[0, 1], [2, 1]])
    radii = np.array([1.0, 0.5, 2.0])
    settled = settle_positions(positions, contacts, radii, 200)
    assert np.linalg.norm(settled[1] - settled[0]) == pytest.approx(1.5, rel=1e-3)
    assert np.linalg.norm(settled[1] - settled[2]) == pytest.approx(2.5, rel=1e-3)
    assert np.array_equal(positions[1], [5.0, 0.0, 0.0])


def test_positions_follow_a_collapse_to_the_energy_weighted_mean_and_a_split_to_near_the_parent():
    def state(kind, energy, contacts):
        return SimpleNamespace(kind=np.array(kind, dtype=np.uint8), energy=np.array(energy, dtype=float),
                               size=np.ones(len(kind)), contacts=np.array(contacts))
    # A0 - B1 - A2 - B3 - A4; joint 1 collapses: A0 and A2 fall into A0 (the lowest), order kept.
    before = state([0, 1, 0, 1, 0], [1.0, 0.05, 3.0, 0.05, 1.0], [[0, 1], [2, 1], [2, 3], [4, 3]])
    after = state([0, 1, 0], [4.05, 0.05, 1.0], [[0, 1], [2, 1]])
    old = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    assert volume_alignment(before.kind, before.energy, after.kind, after.energy).tolist()[1:] == [3, 4]
    carried = carry_positions(before, after, old)
    assert carried[0] == pytest.approx([1.5, 0, 0]) and carried[1:].tolist() == [[3, 0, 0], [4, 0, 0]]
    # A0 (touching B1, B2) splits: A0 keeps B1, the new A3 takes B2, the new joint B4 between them.
    parent = state([0, 1, 1], [8.0, 0.05, 0.06], [[0, 1], [0, 2]])
    split = state([0, 1, 1, 0, 1], [3.9, 0.05, 0.06, 3.9, 0.2], [[0, 1], [0, 4], [3, 2], [3, 4]])
    carried = carry_positions(parent, split, np.array([[0.0, 0, 0], [-1, 0, 0], [1, 0, 0]]))
    assert carried[:3].tolist() == [[0, 0, 0], [-1, 0, 0], [1, 0, 0]]
    assert np.linalg.norm(carried[3]) < 1.5 and np.linalg.norm(carried[4]) < 1.5
    assert not np.allclose(carried[3], carried[4])


def test_the_mesh_key_changes_only_with_a_collapse_or_split_and_positions_stay_finite():
    simulation = Simulation(VOLUMES, seed=3)
    for _ in range(4):
        before_key, before = mesh_key(simulation.state), simulation.state
        simulation.advance()
        events = (simulation.state.collapses - before.collapses) + (simulation.state.splits - before.splits)
        assert (mesh_key(simulation.state) != before_key) == (events > 0)
        assert simulation.positions.shape == (len(simulation.state.kind), 3)
        assert np.all(np.isfinite(simulation.positions))


def test_a_reset_is_reproducible_by_seed_and_settings_are_clamped():
    first, second = Simulation(VOLUMES, seed=4), Simulation(VOLUMES, seed=4)
    for _ in range(2):
        first.advance()
        second.advance()
    assert first.steps == 2
    assert page_frame(first, 1) == page_frame(second, 1) and page_mesh(first, 1) == page_mesh(second, 1)
    first.reset(5, VOLUMES)
    assert first.steps == 0
    first.apply({"push_limit": 5.0})
    assert first.settings.push_limit == first.ranges["push_limit"][1] < 1.0
    with pytest.raises(ValueError):
        first.apply({"no_such_rule": 1.0})


def test_the_settings_message_names_every_rule_then_unevenness_and_speed():
    simulation = Simulation(VOLUMES, seed=1)
    values = settings_message(simulation)["values"]
    assert list(values) == [*simulation.rule_names, "unevenness", "speed"] and simulation.rule_names
    for spec in values.values():
        assert spec["min"] <= spec["value"] <= spec["max"]


def test_the_stats_message_carries_the_engines_stats_and_the_run_state():
    simulation = Simulation(VOLUMES, seed=1)
    message = stats_message(simulation, running=True)
    assert message["type"] == "stats" and message["running"] is True and message["step"] == 0
    assert set(abab.stats(simulation.state, simulation.settings)) == set(message["stats"])
    json.dumps(message, allow_nan=False)


def test_the_socket_greets_then_runs_at_most_ten_frames_a_second_and_answers_a_reset():
    with TestClient(create_app(VOLUMES, seed=1)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "settings"
            mesh = socket.receive_json()
            assert mesh["type"] == "mesh"
            frame = _read_frame(socket.receive_bytes())
            assert frame["mesh_id"] == mesh["mesh_id"] and frame["step"] == 0
            assert len(frame["positions"]) == mesh["count"]
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
            socket.send_json({"type": "reset", "seed": 3, "volumes": VOLUMES})
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


def test_the_page_draws_wire_balls_from_kinds_contacts_and_binary_frames():
    with TestClient(create_app(VOLUMES, seed=1)) as client:
        text = client.get("/").text
    assert "three.js/r128/three.min.js" in text
    assert 'message.type === "mesh"' in text and 'binaryType = "arraybuffer"' in text
    assert f"{FRAME_MAGIC:#x}" == "0x31415251" and "0x31415251" in text
    assert "IcosahedronGeometry" in text and "contactsButton" in text
    assert "#offline[hidden] { display: none; }" in text
    for gone in ("face_cells", "cell_kind", "corner_count"):
        assert gone not in text
