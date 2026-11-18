"""lumps_start, scene_start, volume_energy and lump_path on a real small universe."""
import numpy as np

from physics.engine import JOINT, Settings, book_law, read_books
from physics.scenes.gravity.definitions.lump import Lump
from physics.scenes.gravity.logic.lump_path import lump_path
from physics.scenes.gravity.logic.lumps_start import lumps_start
from physics.scenes.gravity.logic.scene_start import scene_start
from physics.scenes.gravity.logic.volume_energy import volume_energy
from physics.scenes.tests._gpu import needs_gpu

SETTINGS = Settings(friction=0.01, clock_packing=1e12, tear_length=1e6)


@needs_gpu
def test_the_lump_is_where_it_was_put_and_the_start_is_laid_again_at_rest_in_balance():
    universe = lumps_start(SETTINGS, 16, [Lump((20.5, 26.5, 26.5), 4.0, 8.0)])
    energy = volume_energy(universe)
    assert energy.shape == (8, 8, 8) and np.unravel_index(energy.argmax(), energy.shape) == (3, 4, 4)
    assert energy.max() > 5.0 * np.median(energy)
    squeeze = universe.host("squeeze")[universe.host("kind") == JOINT]
    assert squeeze.max() <= 1.0 + 1e-3, "no joint starts squeezed, the lump's own included"
    assert not any(book_law(universe.start_books, read_books(universe)).values())


@needs_gpu
def test_a_kicked_lump_moves_the_way_the_kick_points_and_one_at_rest_stays():
    kicked = lumps_start(SETTINGS, 16, [Lump((20.5, 26.5, 26.5), 4.0, 8.0, (0.6, 0.0, 0.0))])
    still = lumps_start(SETTINGS, 16, [Lump((20.5, 26.5, 26.5), 4.0, 8.0)])
    moved = lump_path(kicked, SETTINGS, [(3.0, 4.0, 4.0)], 1.5, 40, 20)
    stayed = lump_path(still, SETTINGS, [(3.0, 4.0, 4.0)], 1.5, 40, 20)
    assert moved["steps"] == [0, 20, 40] and len(moved["centres"][0]) == 3
    gone = moved["centres"][0][-1][0] - moved["centres"][0][0][0]
    rest = stayed["centres"][0][-1][0] - stayed["centres"][0][0][0]
    print(f"in 40 steps the kicked lump went {gone:+.3f} volumes along the kick, the one at rest {rest:+.3f}")
    assert gone > 0.2 and gone > 4.0 * abs(rest)
    assert moved["collapses"] == 0 and moved["tears"] == 0


@needs_gpu
def test_the_orbit_s_small_lump_gets_its_kick_from_the_pull_and_the_other_scenes_none():
    universe, lumps = scene_start("orbit", SETTINGS, 24)
    assert lumps[0].kick == (0.0, 0.0, 0.0) and lumps[1].kick[0] == 0.0 and 0.0 < lumps[1].kick[1] <= np.pi / 2.0
    assert universe.step == 0 and all(lump.kick == (0.0, 0.0, 0.0) for lump in scene_start("two_lumps", SETTINGS, 24)[1])
