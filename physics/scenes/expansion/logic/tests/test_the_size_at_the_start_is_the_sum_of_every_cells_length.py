"""The scene's size at step 0 is the plain sum of a fresh universe's cell lengths, split into volumes and joints."""
import numpy as np
import pytest

from physics.engine import ENERGY_VOLUME, new_universe
from physics.scenes.expansion.logic.expansion import expansion
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_size_is_summed_lengths_and_the_books_stay_exact():
    result = expansion(8, steps=4, every=2, settings=HELD)
    fresh = new_universe(HELD, 8, seed=1)
    length = fresh.host("length").astype(np.float64)
    volume = fresh.host("kind") == ENERGY_VOLUME
    assert result["size"][0] == pytest.approx(length.sum())
    assert result["volume_size"][0] == pytest.approx(length[volume].sum())
    assert result["joint_size"][0] == pytest.approx(length[~volume].sum())
    assert result["time"] == [0.0, 0.5, 1.0] and result["size_growth"][0] == 1.0
    assert len(result["size_rate"]) == 2 and result["books_gap"] == 0
    assert result["collapses"] == 0 and result["tears"] == 0, "the held grid stands for four steps"
    with pytest.raises(ValueError):
        expansion(8, steps=0, settings=HELD)
