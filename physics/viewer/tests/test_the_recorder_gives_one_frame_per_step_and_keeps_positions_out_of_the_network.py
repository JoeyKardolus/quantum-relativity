"""The recorder returns frame 0 plus one frame per step, and positions never reach a network."""
from dataclasses import fields

import numpy as np

from physics.network import Network, spatial_reference
from physics.rules import RuleSettings
from physics.viewer.record import record
from physics.viewer.scene import Scene


def test_the_recorder_gives_one_frame_per_step_and_keeps_positions_out_of_the_network():
    network, _ = spatial_reference(200, 3, seed=3)
    before = network.energy.copy()
    scene = Scene("tiny", "Tiny", network, RuleSettings(energy_length=1.0, drift=0.01), (), (0, 5),
                  "nothing", seed=3)
    recording = record(scene, 4)
    assert recording.stopped is None
    assert [frame.step for frame in recording.frames] == [0, 1, 2, 3, 4]
    assert all(frame.positions.shape == (200, 3) for frame in recording.frames)
    assert {field.name for field in fields(Network)} <= {
        "joint_count", "pairs", "energy", "length", "charge", "causal", "merged_into"}
    assert np.array_equal(scene.network.energy, before)
    assert recording.frames[-1].mean_length > recording.frames[0].mean_length
