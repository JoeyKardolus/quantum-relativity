"""Axiom 7, Quantum: energy split over several routes settles whole in one cell, chosen by each cell's share.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16): each box's
fluid also carries a phase, so waves add and cancel. Moving energy can split over several
routes at once. Every split gets a direct pull between its halves, private to them, and
its own oscillation, a shared state that keeps turning, starting at the split at a speed
nobody can know. At each turn a pointer jumps to one of the halves' cells, with that
cell's share of the energy as its chance. When something reacts with a half, or the
energy loses its momentum, it settles whole in the cell the pointer is on.

Not built (owner, 2026-11-16: test now, engine later), so this test is marked xfail with
reason "not built"; strict, so the day it passes the marker must come off. Staged once
built: a packet of 1000 quanta in one energy volume, split by a joint into two halves
holding 70% and 30% (two cells); a reaction with the 30% half settles it. Over 200 seeds
the packet ends whole in one cell every time (the other half holds none of it), and in
the 70% cell in 0.63 .. 0.77 of the seeds, never by which half reacted.
"""
import pytest

from physics.engine.tests._gpu import needs_gpu

SEEDS = 200


@needs_gpu
@pytest.mark.xfail(reason="not built", strict=True)
def test_a_split_packet_settles_whole_in_one_cell_by_each_cells_share():
    from physics.engine import split_packet, settle  # not built: the split's own oscillation and its pointer

    wins = 0
    for seed in range(SEEDS):
        state, cells = split_packet(seed=seed, quanta=1000, shares=(0.7, 0.3))
        landed = settle(state, react_at=cells[1])
        assert sorted(landed.values()) == [0, 1000]
        wins += landed[cells[0]] == 1000
    print(f"settled in the 70% cell in {wins} of {SEEDS} seeds")
    assert 0.63 <= wins / SEEDS <= 0.77
