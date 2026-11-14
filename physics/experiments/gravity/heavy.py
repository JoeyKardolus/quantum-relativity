"""One heavy lump past the push limit: does joint collapse run away?

Contract: `heavy_lump_run(base, centre, radius, amplitude, settings, steps)`
puts one lump (`add_lumps`) of the given amplitude on `base` and steps the
rules, recording the joint count each step. It stops when the network can no
longer be read (`safe_step`) or fewer than MIN_JOINTS joints are left. Returns
{"amplitude", "joints", "stopped"}; "stopped" is None when every step ran,
else {"step", "reason"}.
"""
from __future__ import annotations

from physics.experiments.follow import safe_step
from physics.experiments.gravity.lumps import add_lumps
from physics.network import Network
from physics.rules import RuleSettings

MIN_JOINTS = 3


def heavy_lump_run(base: Network, centre: int, radius: float, amplitude: float,
                   settings: RuleSettings, steps: int) -> dict:
    """Step a single heavy lump and record how many joints are left each step."""
    network = add_lumps(base, [centre], radius, amplitude)
    history = {"amplitude": amplitude, "joints": [network.joint_count], "stopped": None}
    for index in range(steps):
        stepped, reason = safe_step(network, settings)
        if stepped is None:
            history["stopped"] = {"step": index + 1, "reason": reason}
            break
        network = stepped
        history["joints"].append(network.joint_count)
        if network.joint_count < MIN_JOINTS:
            history["stopped"] = {"step": index + 1, "reason": f"fewer than {MIN_JOINTS} joints left"}
            break
    return history
