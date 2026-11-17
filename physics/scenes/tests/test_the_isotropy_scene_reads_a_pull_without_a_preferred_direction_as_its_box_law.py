"""The isotropy scene's known answer: with crossing off a placed lump keeps its energy, so its pull is one point's pull.

In a 16-place wrapping box (8 volumes across) a 1 / distance pull is itself not the same
at (3, 0, 0) and (2, 2, 1): the box is a cube and its copies pull too (the law's ratio is
about 0.94, not 1). The scene must read the engine's pull there as that law's ratio, and
the lump must still hold its energy, before its numbers with every rule on mean anything.
"""
from dataclasses import replace

from physics.engine import Settings
from physics.scenes import isotropy
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_pull_of_a_lump_that_keeps_its_energy_matches_the_box_law_off_the_axis():
    found = isotropy(16, steps=400, settle=200, pairs=(((3, 0, 0), (2, 2, 1)),), diagonals=((2, 0, 0), (2, 2, 0)),
                     settings=replace(Settings(), crossing_rate=0.0))
    pair = found["pairs"][0]
    print(f"crossing off, 16 places, 400 steps: pull off the axis / on it {pair['pull_ratio']:.4f}, the box law "
          f"{pair['law_pull_ratio']:.4f}; the lump kept {found['kept']:.3f} of its extra energy; nothing arrived: "
          f"{pair['flow_on']}, {pair['flow_off']}")
    assert pair["pull_on"] > 0.0 and pair["pull_off"] > 0.0
    assert abs(pair["pull_ratio"] / pair["law_pull_ratio"] - 1.0) < 0.03
    assert pair["law_pull_ratio"] < 0.97
    assert abs(found["kept"] - 1.0) < 0.01 and pair["flow_on"] == 0.0 and pair["flow_off"] == 0.0
    assert found["diagonals"][1]["pull_ratio"] < 1.0
