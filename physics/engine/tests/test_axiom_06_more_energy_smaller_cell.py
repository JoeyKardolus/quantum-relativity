"""Axiom 06: more energy makes a cell smaller, across the cells at every check.

Bible rule 3 and rule 4 (length = constant / energy). Default universe (seed 1,
Settings(), axiom_run.py), every 10 steps from step 10 to 300 (at step 0 every cell holds
the same 1000 boxes by construction, so no rank order exists): the Spearman rank
correlation of a cell's quanta against its box count is strongly negative, at most
-0.7, among the energy volumes and among the joints, each kind on its own (their sizes
follow one law but the two kinds are not compared with each other). Threshold set
before the run.
"""
from scipy.stats import spearmanr

from physics.engine.tests._gpu import needs_gpu

MOST = -0.7


@needs_gpu
def test_cells_with_more_quanta_hold_fewer_boxes_at_every_check():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    volume_rho, joint_rho = [], []
    for check in run["checks"][1:]:
        found = check["cells"]
        count = found["volumes"]
        volume_rho.append(spearmanr(found["quanta"][:count], found["boxes"][:count])[0])
        joint_rho.append(spearmanr(found["quanta"][count:], found["boxes"][count:])[0])
    print("volumes rho by check: " + " ".join(f"{value:+.2f}" for value in volume_rho))
    print("joints  rho by check: " + " ".join(f"{value:+.2f}" for value in joint_rho))
    print(f"volumes {min(volume_rho):+.2f} .. {max(volume_rho):+.2f}; joints {min(joint_rho):+.2f} .. "
          f"{max(joint_rho):+.2f} (need every one <= {MOST})")
    assert max(volume_rho) <= MOST and max(joint_rho) <= MOST
