"""The distance between two lumps 2 volumes apart is half of each lump, two joints and the volume between; at the start nothing has moved."""
import pytest

from physics.scenes.tests._gpu import HELD, needs_gpu
from physics.scenes.two_lumps.logic.pair_run import pair_run


@needs_gpu
def test_the_start_distance_and_the_first_steps_of_a_pair():
    run = pair_run(12, 2, 4.0, 3, 1, 1, HELD, "cuda")
    print(f"2 volumes apart: {run['distance']:.1f} lengths; tilt {run['tilt']:.3e}, alone {run['tilt_alone']:.3e}; "
          f"kept {run['kept']:.4f}, closing {run['closing']:+.5f}")
    assert run["distance"] > 2.5 + 10.0, "two half lumps of 2.5, two joints as long as the balance makes them, one volume"
    assert run["reach"] == pytest.approx(0.5 * run["distance"], rel=0.03)
    assert run["round"] == pytest.approx(3 * run["distance"] + 15.0, rel=0.03), "six joints, four volumes, two lumps"
    assert 0.9 < run["kept"] <= 1.01, "three steps in, the lumps still hold their energy"
    assert abs(run["closing"]) < 0.5 and abs(run["closing_alone"]) < 0.5
