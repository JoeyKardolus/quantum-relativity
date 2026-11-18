"""The start's volume graph is the 26-neighbour wrapping grid: both dimensions equal the reference's number for number."""
from physics.engine import new_universe
from physics.scenes.dimension.logic.graph_dimension import graph_dimension
from physics.scenes.dimension.logic.space_dimension import space_dimension
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_a_start_of_512_volumes_is_measured_as_its_own_reference_grid():
    result = space_dimension(16, steps=0, sources=4, settings=HELD)
    print(f"512 volumes: ball {result['start']['ball_dimension']:.3f}, spectral {result['start']['spectral_dimension']:.3f} "
          f"(the plain 26-neighbour grid reads the same; true dimension 3)")
    assert result["reference"]["nodes"] == result["start"]["nodes"] == 512
    assert result["reference"]["ball_dimension"] == result["start"]["ball_dimension"]
    assert result["reference"]["spectral_dimension"] == result["start"]["spectral_dimension"]
    assert result["end"] == result["start"], "no step taken"
    assert graph_dimension(new_universe(HELD, 16, seed=1), 4) == result["start"]
