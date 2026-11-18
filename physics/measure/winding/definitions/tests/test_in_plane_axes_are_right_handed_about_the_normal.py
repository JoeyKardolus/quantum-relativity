"""The walk's two axes: across cross up points along the normal."""
import numpy as np
import pytest

from physics.measure.winding.definitions.in_plane import in_plane


@pytest.mark.parametrize("normal, axes", [(0, (1, 2)), (1, (2, 0)), (2, (0, 1))])
def test_the_two_axes_follow_the_normal_in_turn(normal, axes):
    assert in_plane(normal) == axes
    across, up = np.eye(3)[axes[0]], np.eye(3)[axes[1]]
    assert np.cross(across, up).tolist() == np.eye(3)[normal].tolist()


@pytest.mark.parametrize("normal", [-1, 3])
def test_any_other_normal_is_refused(normal):
    with pytest.raises(ValueError):
        in_plane(normal)
