"""A flows from box to box only through joints: it cannot pack into a box short of its joint share.

Hand-built pair of boxes in a row, all rules' transport (clocks 1): the left box holds
A and B at rest share (b_rest per unit of A), both flowing right at 0.5. Into an empty
box both kinds pass, A at the full donor-cell flux. Into a box holding A with no B (A
would touch A) no A passes while B still does; into a box with half its joint share,
half the A passes. The push of the A held back goes to the receiving box: momentum and
energy are exact in every case.
"""
import numpy as np
import pytest

from physics.fluid import QUANTUM, Grid, Settings
from physics.fluid.transport import transport


def _passed(right_a: float, right_b: float) -> tuple[float, float]:
    settings = Settings.all_on()
    a = np.zeros((4, 1, 1))
    b = np.zeros((4, 1, 1))
    a[1, 0, 0], b[1, 0, 0] = 1.0, settings.b_rest
    a[2, 0, 0], b[2, 0, 0] = right_a, right_b
    velocity = np.zeros((4, 1, 1, 3))
    velocity[1, 0, 0, 0] = 0.5
    cube_a, cube_b = np.zeros((4, 4, 4)), np.zeros((4, 4, 4))
    cube_a[:, :1, :1], cube_b[:, :1, :1] = a, b
    cube_velocity = np.zeros((4, 4, 4, 3))
    cube_velocity[:, :1, :1] = velocity
    grid = Grid.from_host(cube_a, cube_b, cube_velocity, cube_velocity, device="cpu")
    grid.start_a_mean = 1.0
    total = int(grid.a_energy.numpy().sum()) + int(grid.b_energy.numpy().sum())
    momentum = grid.a_momentum.numpy()[..., 0].sum() + grid.b_momentum.numpy()[..., 0].sum()
    transport(grid, settings, 0.1)
    assert int(grid.a_energy.numpy().sum()) + int(grid.b_energy.numpy().sum()) == total
    assert grid.a_momentum.numpy()[..., 0].sum() + grid.b_momentum.numpy()[..., 0].sum() == pytest.approx(momentum)
    return ((grid.a_energy.numpy()[2, 0, 0] * QUANTUM - right_a),
            (grid.b_energy.numpy()[2, 0, 0] * QUANTUM - right_b))


def test_a_needs_joints_to_pass_and_never_packs_without_them():
    rest = Settings.all_on().b_rest
    a_free, b_free = _passed(0.0, 0.0)
    assert a_free == pytest.approx(0.05, rel=1e-6)
    assert b_free == pytest.approx(0.05 * rest, rel=1e-6)
    a_blocked, b_blocked = _passed(1.0, 0.0)
    assert a_blocked == 0.0
    assert b_blocked == pytest.approx(0.05 * rest, rel=1e-6)
    a_half, _ = _passed(1.0, 0.5 * rest)
    assert a_half == pytest.approx(0.025, rel=1e-6)
