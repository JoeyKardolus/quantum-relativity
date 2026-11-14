"""Joints are followed through joint collapse, and a breakdown is named, not passed on.

WHAT IT PINS. physics/experiments/follow through `carry_joints` and
`safe_step`: after a step in which joints collapse, every original joint maps
to the joint it merged into; a step whose numbers blow up comes back as None
with a reason instead of an unreadable network.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Composing two
merge maps by hand (old -> middle -> new) gives the expected map; an attraction
of 1e6 shortens every connection by a factor exp(-1e6 x pull), which underflows
to zero length, so the step cannot be read.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. carry_joints returning the step's merged_into
instead of composing it with the earlier map.

GUARDS. v2 results page, 2026-11-14.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.experiments.follow import carry_joints, safe_step
from physics.network import spatial_reference
from physics.rules import RuleSettings


def test_joints_are_followed_through_joint_collapse_and_a_breakdown_is_named() -> None:
    """Composed merge map; huge attraction gives (None, reason)."""
    network = spatial_reference(500, 3, seed=1)[0]
    earlier = np.array([0, 0, 1, 2])
    stepped = replace(network, merged_into=np.array([1, 0, 0, *range(3, network.joint_count)]))
    np.testing.assert_array_equal(carry_joints(earlier, stepped), [1, 1, 0, 0])
    assert carry_joints(earlier, network) is earlier
    broken, reason = safe_step(network, RuleSettings(attraction=1e6))
    assert broken is None and reason
    fine, none = safe_step(network, RuleSettings())
    assert fine is not None and none is None
