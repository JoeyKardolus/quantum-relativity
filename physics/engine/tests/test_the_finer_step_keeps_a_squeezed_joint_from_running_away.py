"""With fluid substeps from the steepest face, the steep push is integrated stably: no joint runs away.

Owner, 2026-11-16 (decision E, numerics, not a rule): the coarse step (one fluid step
of step_time 0.25) does not resolve the steep push, so edge joints spiked and collapse
ran away from about step 110. The default start (new_universe(), seed 1: 7**3 cells of
10**3 boxes) runs 150 steps twice from the same start: with the default settings
(substeps.py cuts each step's fluid part so no same-sign face's phase difference turns
more than max_turn, 0.25 rad, a substep) and with max_turn 0 (the old coarse step).
Every 10 steps the joint boxes' largest squeeze is read. With substeps no joint box
collapses and the largest squeeze stays below the seal (seal_squeeze, 1) at every
reading; the coarse run's numbers are printed beside them, not judged. Criteria
written before the test ran.
"""
from dataclasses import replace

import numpy as np

from physics.engine import Settings, new_universe, step
from physics.engine.tests._gpu import needs_gpu

STEPS = 150


def _run(settings: Settings) -> tuple:
    """Per reading the largest joint squeeze, the collapsed boxes at the end and the mean substeps a step."""
    universe = new_universe(seed=1)
    readings = []
    taken = 0
    for count in range(1, STEPS + 1):
        step(universe, settings)
        taken += int(universe.substeps.numpy().max())
        if count % 10 == 0:
            quanta = universe.quanta.numpy()
            readings.append(float((-quanta[quanta < 0]).max() / settings.joint_rest - 1.0))
    return readings, int(universe.tallies.numpy()[0, 1]), taken / STEPS, int(universe.capped[0])


@needs_gpu
def test_substeps_keep_every_joint_below_the_seal_where_the_coarse_step_runs_away():
    settings = Settings()
    fine = _run(settings)
    coarse = _run(replace(settings, max_turn=0.0))
    print(f"largest squeeze every 10 steps, substeps: {np.round(fine[0], 2).tolist()}; collapsed {fine[1]}, "
          f"mean substeps {fine[2]:.1f}, capped steps {fine[3]}")
    print(f"largest squeeze every 10 steps, coarse:   {np.round(coarse[0], 2).tolist()}; collapsed {coarse[1]}")
    assert fine[1] == 0
    assert max(fine[0]) < settings.seal_squeeze
