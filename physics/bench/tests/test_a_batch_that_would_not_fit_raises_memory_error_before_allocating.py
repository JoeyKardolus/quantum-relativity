"""A batch bigger than the device's free memory raises MemoryError, so a size search counts it as too big.

Two universes sized to need 5% more than the free memory; on a device so large that this
passes the int32 box index, the test skips (that limit raises ValueError instead).
"""
import pytest
import warp as wp

from physics.bench import new_quanta_batch
from physics.bench.new_quanta_batch import BYTES_PER_BOX
from physics.bench.tests._gpu import needs_gpu


@needs_gpu
def test_too_big_raises_memory_error():
    free = wp.get_device("cuda").free_memory
    side = int(round((free * 1.05 / BYTES_PER_BOX / 2) ** (1 / 3))) + 1
    if 2 * side ** 3 >= 2 ** 31:
        pytest.skip("this device's free memory is past the int32 box index")
    with pytest.raises(MemoryError):
        new_quanta_batch(2, side)
