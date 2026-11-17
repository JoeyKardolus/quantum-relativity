"""The dimension scene's plain grid has as many nodes as the start has volumes, and wraps when it does."""
from physics.scenes import space_dimension
from physics.scenes.space_dimension import _plain_grid
from physics.scenes.tests._gpu import needs_gpu


def test_a_wrapping_plain_grid_gives_every_node_six_neighbours():
    low, high, node_count = _plain_grid(4, wrap=True)
    assert node_count == 64 and low.size == 3 * 64
    walled_low, _, _ = _plain_grid(4, wrap=False)
    assert walled_low.size == 3 * 16 * 3


@needs_gpu
def test_a_wrapping_start_is_measured_against_a_wrapping_grid_of_its_own_size():
    result = space_dimension(8, 6, steps=0, sources=4)
    assert result["reference"]["cells"] == result["start"]["cells"] == 64
    assert result["reference"]["ball_dimension"] == result["start"]["ball_dimension"]
    assert result["reference"]["spectral_dimension"] == result["start"]["spectral_dimension"]
