"""Direction codes: 26 steps, each with its opposite, the same on the host and in a kernel."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_3_structure.definitions.direction_codes import (CENTRE, code_of_host, opposite,
                                                                          opposite_host, step_of, step_of_host)

wp.config.quiet = True


@wp.kernel
def _read_codes(opposites: wp.array(dtype=wp.int32), steps: wp.array(dtype=wp.vec3i)):
    code = wp.tid()
    opposites[code] = opposite(code)
    steps[code] = step_of(code)


def test_hand_worked_codes():
    assert code_of_host(-1, -1, -1) == 0
    assert code_of_host(0, 0, 0) == CENTRE == 13
    assert code_of_host(1, 0, 0) == 22
    assert code_of_host(0, 0, 1) == 14
    assert step_of_host(22) == (1, 0, 0)
    assert step_of_host(4) == (-1, 0, 0)
    assert opposite_host(22) == 4


def test_every_code_has_its_step_and_its_opposite():
    seen = set()
    for code in range(27):
        step = step_of_host(code)
        assert code_of_host(*step) == code
        assert step_of_host(opposite_host(code)) == tuple(-part for part in step)
        seen.add(step)
    assert len(seen) == 27


def test_a_step_outside_one_cell_is_refused():
    with pytest.raises(ValueError):
        code_of_host(2, 0, 0)


def test_kernel_and_host_agree():
    opposites = wp.zeros(27, dtype=wp.int32, device="cpu")
    steps = wp.zeros(27, dtype=wp.vec3i, device="cpu")
    wp.launch(_read_codes, dim=27, inputs=[opposites, steps], device="cpu")
    assert opposites.numpy().tolist() == [opposite_host(code) for code in range(27)]
    assert np.array_equal(steps.numpy(), np.array([step_of_host(code) for code in range(27)]))
