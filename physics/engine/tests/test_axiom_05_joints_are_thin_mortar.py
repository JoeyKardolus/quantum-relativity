"""Axiom 05: joints are thin mortar: they hold a small share of the energy and of the space.

Audit answers (owner, 2026-11-16): joints are thin mortar between volumes, holding a
small share of the energy, never cells as big as volumes. Default universe (seed 1,
Settings(), axiom_run.py), every 10 steps for 300 steps: the joints' share of the
quanta and of the occupied boxes are both well under the volumes' share. "Well under"
is read as at most half the volumes' share, so the joints hold at most a third of
either; the threshold was set before the run. The shares are printed.
"""
from physics.engine.tests._gpu import needs_gpu

MOST = 1.0 / 3.0


@needs_gpu
def test_the_joints_hold_a_small_share_of_the_quanta_and_the_boxes_at_every_check():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    quanta_shares, box_shares = [], []
    for check in run["checks"]:
        found = check["cells"]
        count = found["volumes"]
        quanta_shares.append(found["quanta"][count:].sum() / found["quanta"].sum())
        box_shares.append(found["boxes"][count:].sum() / found["boxes"].sum())
    print(f"joint share of quanta: {min(quanta_shares):.3f} .. {max(quanta_shares):.3f}; of boxes: "
          f"{min(box_shares):.3f} .. {max(box_shares):.3f} (allowed at most {MOST:.3f}); "
          f"step 0: {quanta_shares[0]:.3f} quanta, {box_shares[0]:.3f} boxes; step 300: "
          f"{quanta_shares[-1]:.3f} quanta, {box_shares[-1]:.3f} boxes")
    assert max(quanta_shares) <= MOST and max(box_shares) <= MOST
