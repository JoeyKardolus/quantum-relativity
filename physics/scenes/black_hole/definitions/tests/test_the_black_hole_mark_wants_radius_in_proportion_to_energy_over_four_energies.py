"""The mark passes a radius that doubles with the energy over four free readings spanning a factor 8, and nothing less."""
from physics.scenes.along_grid import log_slope
from physics.scenes.black_hole.definitions.pass_mark import passes


def _result(energies, radii, limits=None):
    limits = limits or [0] * len(energies)
    free = [(energy, radius) for energy, radius, limit in zip(energies, radii, limits) if radius > 0 and not limit]
    return {"core_energy": energies, "held_radius": radii, "held_limit": limits,
            "held_power": log_slope([energy for energy, _ in free], [radius for _, radius in free])}


def test_a_horizon_like_region_passes():
    assert passes(_result([2.0, 4.0, 8.0, 16.0], [10.0, 20.0, 40.0, 80.0]))
    assert passes(_result([2.0, 4.0, 8.0, 16.0], [10.0, 21.0, 39.0, 85.0]))


def test_a_shrinking_core_too_few_readings_or_too_narrow_a_span_miss():
    assert not passes(_result([2.0, 4.0, 8.0, 16.0], [80.0, 40.0, 20.0, 10.0])), "power -1: the core's own size"
    assert not passes(_result([2.0, 4.0, 8.0, 16.0], [10.0, 14.0, 20.0, 28.0])), "power 0.5"
    assert not passes(_result([2.0, 4.0, 8.0, 16.0], [0.0, 20.0, 40.0, 80.0])), "three readings"
    assert not passes(_result([2.0, 4.0, 8.0, 16.0], [10.0, 20.0, 40.0, 80.0], [0, 0, 0, 1])), "one at the box's limit"
    assert not passes(_result([2.0, 3.0, 4.0, 6.0], [10.0, 15.0, 20.0, 30.0])), "energies span a factor 3"
