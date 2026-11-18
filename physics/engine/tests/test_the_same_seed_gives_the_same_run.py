"""The same seed gives the same run, bit for bit, through collapses and through tears; another seed does not."""
from __future__ import annotations

import pytest

from physics.engine import Settings, new_universe, step
from physics.engine.tests.conftest import needs_gpu, state_digest

pytestmark = needs_gpu


def _run(changed: dict, seed: int, steps: int):
    """A small universe run for some steps: its digest and its counts of events."""
    settings = Settings(**changed)
    state = new_universe(settings, cells_per_side=8, seed=seed, device="cuda")
    for _ in range(steps):
        step(state, settings)
    return state_digest(state), int(state.collapses), int(state.tears)


@pytest.mark.parametrize("changed, steps", [({"collapse_thin": 1.01}, 40), ({"unevenness": 0.6}, 6)])
def test_two_runs_from_one_seed_end_with_the_same_bits(changed, steps):
    first, second = _run(changed, 1, steps), _run(changed, 1, steps)
    assert first == second
    assert first[1] + first[2] > 0, "the run had no event, so events were not compared"


def test_another_seed_gives_another_run():
    assert _run({}, 1, 5)[0] != _run({}, 2, 5)[0]
