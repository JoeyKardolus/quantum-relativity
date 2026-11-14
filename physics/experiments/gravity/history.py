"""Step the rules on a lumped network and track the two lumps each step.

Contract: `lump_history(network, settings, joints, far_joints, steps)` steps
the rules `steps` times and returns (records, snapshots, stopped): one
`track_lumps` record per readable step (step 0 included) and, for the same
steps, (network, carried) with carried[original joint] = its index in that
network. Tracked joints and the far-background joints follow joint collapse
through merged_into; each record also carries "joints" (joint count) and
"origin": per lump centre, the original joints now merged into it (for the
fixed ruler). The run stops early, with stopped = {"step", "reason"}, when
`safe_step` reports the network unreadable (else None).
"""
from __future__ import annotations

import numpy as np

from physics.experiments.follow import carry_joints, safe_step
from physics.experiments.gravity.tracking import track_lumps
from physics.network import Network
from physics.rules import RuleSettings


def lump_history(network: Network, settings: RuleSettings, joints: tuple[int, int],
                 far_joints: np.ndarray, steps: int):
    """Records per step, (network, carried) per step, stop note."""
    carried = np.arange(network.joint_count)
    records = [_record(network, carried, joints, far_joints)]
    snapshots = [(network, carried)]
    stopped = None
    for index in range(steps):
        stepped, reason = safe_step(network, settings)
        if stepped is None:
            stopped = {"step": index + 1, "reason": reason}
            break
        network, carried = stepped, carry_joints(carried, stepped)
        records.append(_record(network, carried, joints, far_joints))
        snapshots.append((network, carried))
    return records, snapshots, stopped


def _record(network: Network, carried: np.ndarray, joints: tuple[int, int],
            far_joints: np.ndarray) -> dict:
    now = (int(carried[joints[0]]), int(carried[joints[1]]))
    record = track_lumps(network, now, np.unique(carried[far_joints]))
    record["joints"] = network.joint_count
    record["origin"] = [np.flatnonzero(carried == lump["centre"]) for lump in record["lumps"]]
    return record
