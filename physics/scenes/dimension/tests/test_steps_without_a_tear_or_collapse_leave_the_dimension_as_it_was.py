"""A cell's neighbours change only at a tear or a collapse, so two steps of a standing grid leave both dimensions unchanged."""
from physics.scenes.dimension import KNOWN_DIMENSION, passes, space_dimension
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_two_steps_of_a_held_grid_change_no_relation_and_so_no_dimension():
    result = space_dimension(16, steps=2, sources=4, settings=HELD)
    assert result["collapses"] == 0 and result["tears"] == 0
    assert result["end"] == result["start"]
    assert result["known_dimension"] == KNOWN_DIMENSION
    both = all(2.7 <= result["end"][name] <= 3.3 for name in ("ball_dimension", "spectral_dimension"))
    assert passes(result) == both
