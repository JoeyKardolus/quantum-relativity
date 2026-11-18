"""Size law: length = constant / quanta, one constant for volumes and joints, an empty cell endless."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, ENERGY_VOLUME, JOINT, NO_CELL
from physics.engine.axiom_3_structure.definitions.size_law import size_law, size_law_host
from physics.engine.axiom_3_structure.definitions.structure_settings import StructureSettings

wp.config.quiet = True


@wp.kernel
def _sizes(cell_quanta: wp.array(dtype=wp.int64), size_constant: wp.float64, length: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    length[cell] = size_law(cell_quanta[cell], size_constant)


def test_kinds_and_block():
    assert (ENERGY_VOLUME, JOINT, NO_CELL) == (1, -1, 0)
    assert BLOCK == 6 and BOXES == 216


def test_hand_worked_lengths():
    constant = StructureSettings().size_constant
    assert constant == 6.48e6
    assert size_law_host(3000 * 216, constant) == 10.0      # a start volume
    assert size_law_host(100 * 216, constant) == 300.0      # a start joint, same constant
    assert size_law_host(2 * 3000 * 216, constant) == 5.0   # twice the energy, half the length
    assert math.isinf(size_law_host(0, constant))           # an empty cell is endlessly long


def test_kernel_and_host_agree():
    quanta = np.array([0, 1, 3, 21600, 648000, 648001, 5_000_000_000], dtype=np.int64)
    length = wp.zeros(len(quanta), dtype=wp.float32, device="cpu")
    wp.launch(_sizes, dim=len(quanta), inputs=[wp.array(quanta, dtype=wp.int64, device="cpu"), 6.48e6, length],
              device="cpu")
    assert length.numpy().tolist() == [size_law_host(int(one), 6.48e6) for one in quanta]
    assert length.numpy()[4] > length.numpy()[5]            # more energy, smaller


def test_the_constant_must_be_above_zero():
    for wrong in (0.0, -1.0):
        with pytest.raises(ValueError):
            StructureSettings(size_constant=wrong)
    with pytest.raises(TypeError):
        StructureSettings(start_length=1.0)
