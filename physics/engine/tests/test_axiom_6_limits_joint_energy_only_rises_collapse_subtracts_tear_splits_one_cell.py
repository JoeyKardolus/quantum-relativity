"""Axiom 6, Limits: a joint's energy rises only while it is squeezed; squeezed too far it collapses, its energy is taken off the rest and its two volumes merge; an overstretched volume tears, and only that cell splits.

Checked at every step of the default run (1,000 steps) and the eventful run (60 steps,
hundreds of tears and collapses), axiom_run.py:
  - no joint's quanta ever fall; a joint's quanta rise only in a step in which its
    squeeze is above 1, and (in steps without events) by exactly what it made
Staged, 8 places a side:
  - collapse: a joint between two volumes of 5 times the energy thins step by step as
    it is squeezed (squeeze_rate 0.01 so the thinning takes several steps; at the
    default it takes one) and collapses: it and one volume are no cell any more, the other holds both
    volumes' quanta less the joint's, the total falls by twice the joint's quanta, no
    other cell's quanta change in that step's collapse, the merged volume's neighbours
    are the two volumes' neighbours without the joint, and the pattern holds
  - tear: a volume with 0.4 of the energy (2.5 times as long, past the tear length)
    tears in the first step: two new ids, a joint and a volume; the three cells hold
    exactly what the one held, the joint what turned from plus to minus; no other
    cell's quanta, kind or neighbours change except that the torn cell's own neighbours
    now list the halves; the pattern holds
  - a fresh joint from a tear is not squeezed and makes nothing
"""
import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, Settings, new_universe, pattern, step, totals
from physics.engine.collapse import collapse
from physics.engine.tear import tear
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run

_BREAKS = ("touching", "unwrapped", "one_sided", "dead_links", "loose_ways")
JOINT_ID, LOW, HIGH = (1 * 8 + 2) * 8 + 2, (0 * 8 + 2) * 8 + 2, (2 * 8 + 2) * 8 + 2


@needs_gpu
def test_a_joints_energy_never_falls_and_rises_only_while_squeezed_at_every_step():
    for name in ("default", "eventful"):
        steps = axiom_run(name)["steps"]
        fell = sum(row["joint_fell"] for row in steps)
        rose = sum(row["joint_rose"] for row in steps)
        loose = sum(row["joint_rose_loose"] for row in steps)
        wrong = sum(row["joint_gain_wrong"] for row in steps)
        print(f"{name}: joint-steps in which a joint's quanta fell {fell}, rose {rose}, rose without being squeezed "
              f"{loose}, rose by something other than what it made {wrong}; made in all {steps[-1]['made']}")
        assert fell == 0 and loose == 0 and wrong == 0
    assert sum(row["joint_rose"] for row in axiom_run("eventful")["steps"]) > 0


@needs_gpu
def test_a_joint_squeezed_too_far_collapses_and_its_energy_is_taken_off_its_merged_volumes():
    state, settings = new_universe(8, unevenness=0.0), Settings(crossing_rate=0.0, squeeze_rate=0.01)
    for volume in (LOW, HIGH):
        state.fill(volume, 5 * 648000)
    around = set(state.neighbours(LOW)[0].tolist()) | set(state.neighbours(HIGH)[0].tolist())
    thins = []
    for _ in range(400):
        held, books = state.host("cell_quanta").copy(), totals(state)
        step(state, settings)
        thins.append(int(held[JOINT_ID]))
        if state.collapses:
            break
    now, after = state.host("cell_quanta"), totals(state)
    joint_quanta = after["taken_joint"]
    others = np.ones(state.count, dtype=bool)
    others[[JOINT_ID, LOW, HIGH]] = False
    made = now[others] - held[others]
    found = pattern(state)
    print(f"joint between two volumes of 5 times the energy: quanta {thins[0]} -> {thins[-1]} over {len(thins)} steps, "
          f"then collapse with {joint_quanta}; merged volume {int(now[LOW])} = {int(held[LOW])} + {int(held[HIGH])} - "
          f"{joint_quanta}; total {books['amount']} -> {after['amount']} with {after['made'] - books['made']} made that "
          f"step; pattern {found}")
    assert state.collapses == 1 and state.merges == 1 and state.tears == 0
    assert all(later >= earlier for earlier, later in zip(thins, thins[1:])) and thins[-1] > thins[0]
    assert joint_quanta >= settings.collapse_thin * state.start_quanta[JOINT]
    assert state.kind_h[JOINT_ID] == 0 and state.kind_h[HIGH] == 0 and state.kind_h[LOW] == ENERGY_VOLUME
    assert now[JOINT_ID] == 0 and now[HIGH] == 0
    assert now[LOW] == held[LOW] + held[HIGH] - joint_quanta and after["taken_volumes"] == joint_quanta
    assert after["amount"] - books["amount"] == (after["made"] - books["made"]) - 2 * joint_quanta
    assert int(made.sum()) == after["made"] - books["made"] - (joint_quanta - int(held[JOINT_ID])) and (made >= 0).all()
    assert set(state.neighbours(LOW)[0].tolist()) == around - {JOINT_ID, LOW, HIGH}
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)
    assert found["volumes"] == 63 and found["joints"] == 447


@needs_gpu
def test_an_overstretched_volume_tears_and_only_that_cell_splits():
    state, settings = new_universe(8, unevenness=0.0), Settings(crossing_rate=0.0)
    torn = (4 * 8 + 4) * 8 + 4
    state.fill(torn, int(0.4 * 648000))
    held, kinds, books = state.host("cell_quanta").copy(), state.kind_h[:state.count].copy(), totals(state)
    before = {cell: set(state.neighbours(cell)[0].tolist()) for cell in range(state.count)}
    beside = before[torn]
    step(state, settings)
    now, after = state.host("cell_quanta"), totals(state)
    joint, twin = 512, 513
    found = pattern(state)
    changed = [cell for cell in range(512) if cell != torn
               and set(state.neighbours(cell)[0].tolist()) - {torn, joint, twin} != before[cell] - {torn}]
    grew = [cell for cell in range(512) if cell != torn and cell not in beside
            and set(state.neighbours(cell)[0].tolist()) != before[cell]]
    print(f"a volume of 0.4 of the energy ({int(held[torn])} quanta, length {2.5 * 10:.0f}) tore in step 1: it keeps "
          f"{int(now[torn])}, the new joint {joint} holds {int(now[joint])} (turned minus: {after['torn']}), the new "
          f"volume {twin} holds {int(now[twin])}; other cells whose quanta changed "
          f"{int((now[:512] != held)[np.arange(512) != torn].sum())}; neighbours changed beyond listing the halves "
          f"{len(changed)}; cells not beside it whose neighbours changed {len(grew)}; pattern {found}; squeeze of the "
          f"new joint {float(state.host('squeeze')[joint]):.2e}")
    assert state.tears == 1 and state.count == 514 and state.collapses == 0
    assert state.kind_h[joint] == JOINT and state.kind_h[twin] == ENERGY_VOLUME and state.kind_h[torn] == ENERGY_VOLUME
    assert now[torn] + now[joint] + now[twin] == held[torn] and after["torn"] == now[joint]
    assert after["amount"] == books["amount"] and after["plus"] == books["plus"] - now[joint]
    assert np.array_equal(now[:512][np.arange(512) != torn], held[np.arange(512) != torn])
    assert np.array_equal(state.kind_h[:512], kinds) and not changed and not grew
    assert state.ways(joint) == [(torn, twin)]
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)
    assert found["volumes"] == 65 and found["joints"] == 449
    step(state, settings)
    assert state.host("cell_quanta")[joint] == now[joint] and state.host("squeeze")[joint] <= 1.0


@needs_gpu
def test_a_torn_volume_that_falls_back_together_has_its_old_neighbours_again():
    state = new_universe(8, unevenness=0.0)
    torn = (4 * 8 + 4) * 8 + 4
    state.fill(torn, int(0.4 * 648000))
    before = {cell: set(state.neighbours(cell)[0].tolist()) for cell in range(512)}
    held = {name: getattr(state, name).numpy() for name in ("cell_quanta", "pull", "pull_old", "clock", "jiggle")}
    assert tear(state, held, torn, 20.0) and collapse(state, held, 512)
    after = {cell: set(state.neighbours(cell)[0].tolist()) for cell in range(512)}
    found = pattern(state)
    print(f"tear then collapse of the new joint: cells whose neighbours differ from the start "
          f"{sum(after[cell] != before[cell] for cell in range(512))}; pattern {found}")
    assert after == before and found["volumes"] == 64 and found["joints"] == 448
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)


@needs_gpu
def test_a_volume_with_nearly_empty_boxes_still_tears_into_thirds_exactly():
    state = new_universe(8, unevenness=0.0)
    torn, amount = (4 * 8 + 4) * 8 + 4, 100000
    state.fill(torn, amount)
    lopsided = np.ones((6, 6, 6), dtype=np.int32)
    lopsided[0, 0, 0] = amount - 215
    state.set_block(torn, lopsided)
    held = {name: getattr(state, name).numpy() for name in ("cell_quanta", "pull", "pull_old", "clock", "jiggle")}
    assert tear(state, held, torn, 20.0)
    parts = [int(np.abs(state.block(cell)[0]).sum()) for cell in (torn, 512, 513)]
    least = min(int(np.abs(state.block(cell)[0]).min()) for cell in (torn, 512, 513))
    print(f"a volume of {amount} quanta with all but 215 in one box tore into {parts}; fewest quanta in a box {least}")
    assert sum(parts) == amount and max(parts) - min(parts) <= 1 and least >= 1
    assert [int(held["cell_quanta"][cell]) for cell in (torn, 512, 513)] == parts
    assert not tear(state, held, 513, 1e12) and state.tears == 1
