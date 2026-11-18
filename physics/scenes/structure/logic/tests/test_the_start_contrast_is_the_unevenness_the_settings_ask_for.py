"""At step 0 the volumes' contrast is the start's unevenness: 0.1 by default, three times that when a start asks 0.3."""
import pytest

from physics.scenes.structure.logic.structure import structure
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_contrast_at_step_0_is_the_settings_unevenness():
    plain = structure(8, steps=2, every=2, settings=HELD)
    rough = structure(8, steps=2, every=2, settings=HELD, start={"unevenness": 0.3})
    print(f"contrast at step 0: {plain['rms_contrast'][0]:.3f} default, {rough['rms_contrast'][0]:.3f} with unevenness 0.3")
    assert plain["rms_contrast"][0] == pytest.approx(0.1, rel=0.15)
    assert rough["rms_contrast"][0] == pytest.approx(0.3, rel=0.2)
    assert plain["wavenumbers"] == [1.0, 2.0] and len(plain["power"]) == 2 and plain["time"] == [0.0, 0.5]
    with pytest.raises(ValueError):
        structure(4, steps=1, settings=HELD)
