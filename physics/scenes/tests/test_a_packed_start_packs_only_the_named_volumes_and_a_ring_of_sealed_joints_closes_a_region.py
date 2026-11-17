"""A placed start packs exactly the named volumes with exact books; a region counts as sealed only when sealed joints close it off."""
import numpy as np
import pytest

from physics.engine import JOINT, Settings, cell_laws, stats, way_list
from physics.scenes import packed_start, sealed_region, volume_field
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_only_the_named_volumes_are_packed_and_the_books_start_exact():
    universe = packed_start(8, {(4, 4, 4): 4.0, (0, 2, 6): 2.0})
    field, left_out = volume_field(universe)
    mean = universe.start_quanta[1]
    assert left_out == 0 and field.shape == (4, 4, 4)
    assert field[2, 2, 2] == round(4.0 * mean) and field[0, 1, 3] == round(2.0 * mean)
    rest = np.delete(field.reshape(-1), [2 * 16 + 2 * 4 + 2, 0 * 16 + 1 * 4 + 3])
    assert np.abs(rest / mean - 1.0).max() < 0.02, "every other volume is the even start's"
    now = stats(universe)
    assert now["books_gap"] == 0 and now["wrong_cells"] == 0 and now["lumps"] == 2
    length = universe.host("length")
    assert length[(4 * 8 + 4) * 8 + 4] == pytest.approx(10.0 / 4.0, rel=1e-3), "four times the energy, a quarter the length"
    with pytest.raises(ValueError):
        packed_start(8, {(1, 0, 0): 2.0})


@needs_gpu
def test_a_region_is_sealed_only_when_every_joint_out_of_it_is_sealed():
    settings = Settings()
    universe = packed_start(8, {(4, 4, 4): 4.0})
    core = (4 * 8 + 4) * 8 + 4
    open_region = sealed_region(universe, settings)
    assert open_region["core"] == core and open_region["core_joints"] == 6
    assert open_region["sealed"] == 0 and open_region["core_sealed"] == 0 and open_region["radius"] == 0.0
    joints, first, second = way_list(universe)
    around = joints[(first == core) | (second == core)]
    thick = int(settings.seal_thin * universe.start_quanta[JOINT]) + 216
    for joint in around[:5]:
        universe.fill(int(joint), thick)
    cell_laws(universe, settings)
    assert sealed_region(universe, settings)["sealed"] == 0, "one open joint and the region is open"
    universe.fill(int(around[5]), thick)
    cell_laws(universe, settings)
    closed = sealed_region(universe, settings)
    assert closed["sealed"] == 1 and closed["core_sealed"] == 6 and closed["volumes"] == 1
    assert closed["energy"] == closed["core_quanta"] and closed["radius"] == pytest.approx(0.5 * 2.5, rel=1e-3)
