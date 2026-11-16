"""Axiom 01: energy is whole signed quanta per Planck box, and the total ignoring sign is exactly conserved.

Bible rule 4. The default universe (new_universe(seed=1), Settings(), axiom_run.py) runs
300 steps; after every step the boxes hold int32 whole numbers and the sum of |quanta|
equals the start's to the quantum.
"""
import numpy as np

from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_the_amount_ignoring_sign_is_the_same_whole_number_after_every_step():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    start = run["start"]["amount"]
    drifts = np.array([row["amount"] - start for row in run["steps"]])
    print(f"start amount {start}; steps {len(drifts)}; largest drift {np.abs(drifts).max()} quanta")
    assert len(drifts) >= 300
    assert not drifts.any()
