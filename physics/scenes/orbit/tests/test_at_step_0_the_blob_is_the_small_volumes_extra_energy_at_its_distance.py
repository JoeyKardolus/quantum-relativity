"""At step 0 the blob is one even volume's worth of extra energy, exactly two volumes from the big one; a short run makes no whole turn."""
import pytest

from physics.scenes.orbit import orbit, passes
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_start_of_an_orbit_run_and_a_few_steps_of_it():
    result = orbit(16, distances=(2,), steps=3, every=1, settings=HELD)
    run = result["runs"][0]
    print(f"3 steps: the blob keeps {run['extra'][-1] / run['extra'][0]:.3f} of its extra energy, "
          f"centre at {run['radius'][-1]:.3f} volumes, turned {run['turned']:.4f}")
    assert run["extra"][0] == pytest.approx(648000.0, rel=0.03), "2 times an even volume, less the twin's even one"
    assert run["radius"][0] == pytest.approx(2.0)
    assert run["time"] == [0.0, 0.25, 0.5, 0.75] and len(run["radius"]) == 4
    assert abs(run["turned"]) < 1.0 and not passes(result)
    assert result["sideways"] == pytest.approx(0.5235987755982988)
