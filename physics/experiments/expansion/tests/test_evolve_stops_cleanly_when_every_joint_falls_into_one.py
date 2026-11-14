"""Evolve stops cleanly, with a reason, when joint collapse leaves no connection.

WHAT IT PINS. physics/experiments/expansion through `evolve`: a run whose push
limit every joint is past ends with a breakdown step and reason instead of
raising, its sparse joint count per step falls, and the last measurement is of
a network that could still be read.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. With a push
limit of 1e-12 every joint with any pull is asked for more than the limit on
the first step and gives way, so the network keeps falling together until no
connection is left or a connection is too short to read; either way the run
cannot reach its last step.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference
from physics.experiments.expansion.evolve import evolve
from physics.rules import all_rules_on


def test_evolve_stops_cleanly_when_every_joint_falls_into_one() -> None:
    """Breakdown recorded, sparse count falls, every recorded scale read before the end."""
    network, labels = clustered_reference(300, 1, 60, 0.03, 6, seed=4)
    sparse = np.flatnonzero(labels == -1)
    probes = {"sparse_sources": sparse[:2], "pair_rows": np.array([0, 1]),
              "pair_targets": sparse[2:4], "cluster_sources": [np.flatnonzero(labels == 0)[:2]]}
    settings = all_rules_on(attraction=1e-3, push_limit=1e-12)
    record = evolve(network, labels, settings, steps=40, every=5, probes=probes)
    assert record["breakdown_step"] is not None and record["breakdown_step"] < 40
    assert record["breakdown_reason"]
    assert record["sparse_by_step"][0] == 300
    assert record["sparse_by_step"][-1] < 300
    assert len(record["sparse_by_step"]) == record["breakdown_step"]
