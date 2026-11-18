"""On a held grid every volume has an open split after three steps; a detector on each settles it whole, by reaction, with exact books."""
import pytest

from physics.scenes.quantum_split.logic.quantum_split import quantum_split
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_sixty_four_splits_open_by_themselves_and_all_settle_whole():
    found = quantum_split(8, wait=3, settings=HELD)
    print(f"8 a side: {found['splits']} splits over {found['halves']:.1f} cells each; {found['settled']} settled, "
          f"{found['whole_share']:.3f} whole; in the largest half {found['found_in_largest']:.3f}, expected "
          f"{found['expected_in_largest']:.3f}")
    assert found["splits"] == 64, "one split per volume: every volume's outflow met a fork"
    assert found["halves"] > 2.0, "a split lies in several cells at once"
    assert found["settled"] == found["judged"] == 64 and found["unread"] == 0
    assert found["by_reaction"] == 64 and found["by_momentum"] == 0
    assert found["whole_share"] == 1.0 and found["books_gap"] == 0
    assert 0.0 < found["expected_in_largest"] <= 1.0
    with pytest.raises(ValueError):
        quantum_split(8, wait=0, settings=HELD)
