"""Total energy is exact over 300 steps of the default start while friction spreads energy round the seams.

new_universe() (7 cells of 10 boxes a side, seed 1, unevenness 0.3), Settings.all_on(),
300 steps: friction spreads the energy it takes (spread_heat.py) and the sum of
|psi|**2 moves by float64 rounding only. The run's growth, lumps, sealed lumps and
collapses are printed beside it, not asserted (rule outcomes are measured, never
tuned). About a minute on the RTX 5080.
"""
from physics.potts import Settings, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_total_energy_moves_by_rounding_only_over_300_steps_with_spreading():
    universe = new_universe()
    settings = Settings.all_on()
    for count in range(300):
        step(universe, settings)
        if (count + 1) % 50 == 0:
            now = stats(universe, settings)
            print(f"step {count + 1}: occupied x{now['occupied_growth']:.3f}, spread {now['spread']:.0f}, "
                  f"lumps {now['lumps']}, sealed {now['sealed_lumps']}, collapses {now['collapses']}, "
                  f"drift {now['relative_drift']:.1e}")
    end = stats(universe, settings)
    assert end["spread"] > 0
    assert abs(end["relative_drift"]) < 1e-10
