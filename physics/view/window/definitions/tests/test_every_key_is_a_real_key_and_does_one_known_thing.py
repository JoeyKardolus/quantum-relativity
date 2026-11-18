"""The key table, the snapshot and the window's fixed numbers."""
import dataclasses

import numpy as np
import pytest

from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS
from physics.view.window.definitions.key_actions import KEY_ACTIONS
from physics.view.window.definitions.snapshot import Snapshot
from physics.view.window.definitions.window_marks import (HEIGHT, IDLE_WAIT, PITCH_LIMIT, STATS_EVERY, TITLE_EVERY,
                                                          TURN_PER_PIXEL, WIDTH, ZOOM_PER_NOTCH)


def test_the_keys_the_owner_uses_do_what_the_overlay_says():
    assert KEY_ACTIONS["C"] == ("whole", None), "C switches between the whole grid and the slice"
    assert KEY_ACTIONS["SPACE"] == ("pause", None) and KEY_ACTIONS["R"] == ("reset", None)
    assert KEY_ACTIONS["F"] == ("fit", None) and KEY_ACTIONS["ESCAPE"] == ("close", None)
    assert [KEY_ACTIONS[name][1] for name in ("_1", "_2", "_3")] == [SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH]
    assert all(KEY_ACTIONS[f"NUM_{digit}"] == KEY_ACTIONS[f"_{digit}"] for digit in "123")
    assert {KEY_ACTIONS[name] for name in ("UP", "RIGHT", "BRACKETRIGHT")} == {("move", 1)}
    assert {KEY_ACTIONS[name] for name in ("DOWN", "LEFT", "BRACKETLEFT")} == {("move", -1)}
    assert {action for action, _ in KEY_ACTIONS.values()} == {"pause", "whole", "show", "move", "fit", "reset", "close"}


def test_every_key_name_is_a_key_pyglet_knows_and_no_two_names_are_one_key():
    key = pytest.importorskip("pyglet.window.key")
    symbols = [getattr(key, name) for name in KEY_ACTIONS]
    assert len(set(symbols)) == len(KEY_ACTIONS) == 17


def test_a_snapshot_cannot_be_changed_once_published():
    snapshot = Snapshot(3, np.zeros((0, 8, 3)), np.zeros((0, 4)), np.zeros(0), 0, np.zeros(3), np.ones(3), {}, 0.0,
                        True, "", 0, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.step = 4
    assert dataclasses.replace(snapshot, error="broke").error == "broke" and snapshot.error == ""


def test_the_marks_keep_up_up_and_zoom_inward():
    assert 0 < PITCH_LIMIT < np.pi / 2 and 0 < ZOOM_PER_NOTCH < 1 and TURN_PER_PIXEL > 0
    assert (WIDTH, HEIGHT) == (1280, 800) and 0 < IDLE_WAIT < TITLE_EVERY < STATS_EVERY <= 1.0
