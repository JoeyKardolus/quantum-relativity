"""Axiom 3, Structure: the cells are the grid, a perfect A B A B pattern; a cell has only relations and a size; boxes are not space.

The owner's axiom (corrected 2026-11-17): plus never touches plus; minus wraps every
plus and forms one web; the pattern never breaks and never flickers; a cell has no
geometry of its own, only relations to its neighbours and a size set by its energy
alone; Planck boxes sit inside each cell only to compute its fluid. Checked on the
default run (1,000 steps) and the eventful run (60 steps, hundreds of tears and
collapses), axiom_run.py:
  - the start: every energy volume has exactly 26 neighbours, all joints, one web
  - the pattern at step 0, every 100 steps and after every step with a tear or a
    collapse: no volume touches anything but living joints, every volume has a joint in
    all 26 directions, every relation is answered, the joints are one web
  - a cell's neighbours change only at a tear or a collapse: steps in which the
    relations on the GPU changed without one: 0
  - size is exactly the law of the cell's quanta, float32(constant / quanta), for every
    cell at every step
  - no box ever leaves its cell: every living cell's own boxes sum to its quanta, an id
    that stopped being a cell is never a cell again and its boxes stay empty
  - lumps warp the grid: the volume with the most quanta is the shortest
"""
import numpy as np

from physics.engine import BLOCK, ENERGY_VOLUME, JOINT, Settings, new_universe, pattern, step, totals
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run

_BREAKS = ("touching", "unwrapped", "one_sided", "dead_links", "loose_ways")


@needs_gpu
def test_the_start_is_the_perfect_pattern():
    state = new_universe(8)
    found = pattern(state)
    counts = [len(set(state.neighbours(cell)[0].tolist())) for cell in np.flatnonzero(state.kind_h == ENERGY_VOLUME)]
    kinds = {int(state.kind_h[other]) for cell in np.flatnonzero(state.kind_h == ENERGY_VOLUME)
             for other in state.neighbours(cell)[0]}
    print(f"8 places a side: {found['volumes']} volumes, {found['joints']} joints, neighbours per volume "
          f"{min(counts)} .. {max(counts)}, kinds beside a volume {kinds}, web pieces {found['web_pieces']}")
    assert found["volumes"] == 64 and found["joints"] == 448
    assert min(counts) == max(counts) == 26 and kinds == {JOINT}
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)


@needs_gpu
def test_an_uneven_start_of_the_joints_changes_only_their_amounts():
    even, uneven = new_universe(8, seed=3), new_universe(8, seed=3, joint_unevenness=0.4)
    joints = even.kind_h[:even.count] == JOINT
    plain, spread = even.host("cell_quanta"), uneven.host("cell_quanta")
    found = pattern(uneven)
    law = uneven.size_energy[JOINT] / spread[joints].astype(np.float64)
    print(f"joints' quanta spread (sd / mean) {plain[joints].std() / plain[joints].mean():.4f} even, "
          f"{spread[joints].std() / spread[joints].mean():.4f} at joint_unevenness 0.4")
    assert np.array_equal(plain[~joints], spread[~joints]), "the volumes are the same start"
    assert spread[joints].std() / spread[joints].mean() > 0.3 > 0.02 > plain[joints].std() / plain[joints].mean()
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)
    assert np.array_equal(uneven.host("length")[joints], law.astype(np.float32)), "size is still the law of the quanta"


@needs_gpu
def test_the_pattern_never_breaks_in_either_run():
    for name in ("default", "eventful"):
        run = axiom_run(name)
        breaks = {key: sum(found[key] for _, found in run["patterns"]) for key in _BREAKS}
        webs = {found["web_pieces"] for _, found in run["patterns"]}
        last = run["patterns"][-1][1]
        print(f"{name}: pattern checked {len(run['patterns'])} times; breaks summed {breaks}; web pieces seen {webs}; "
              f"at the end {last['volumes']} volumes, {last['joints']} joints, {last['neighbours']:.1f} joints beside "
              f"a volume on average; collapses {run['final']['collapses']}, tears {run['final']['tears']}")
        assert not any(breaks.values()) and webs == {1}
    assert len(axiom_run("eventful")["patterns"]) > 10


@needs_gpu
def test_neighbours_change_only_at_a_tear_or_a_collapse():
    for name in ("default", "eventful"):
        steps = axiom_run(name)["steps"]
        loose = sum(1 for row in steps if row["relinked"] and not row["events"])
        with_events = sum(1 for row in steps if row["events"])
        print(f"{name}: steps with a tear or collapse {with_events}; steps in which a relation changed without one: "
              f"{loose}; kinds on the GPU that differ from the host's, summed: {sum(row['kind_wrong'] for row in steps)}")
        assert loose == 0 and sum(row["kind_wrong"] for row in steps) == 0


@needs_gpu
def test_size_is_exactly_the_law_of_the_cells_quanta_at_every_step():
    for name in ("default", "eventful"):
        steps = axiom_run(name)["steps"]
        wrong = sum(row["size_wrong"] for row in steps)
        shortest = sum(row["quanta_max_is_shortest"] for row in steps)
        print(f"{name}: cells whose length is not float32(constant / quanta), summed over {len(steps)} steps: {wrong}; "
              f"steps in which the volume with the most quanta is the shortest: {shortest}")
        assert wrong == 0 and shortest == len(steps)


@needs_gpu
def test_no_box_ever_leaves_its_cell():
    for name in ("default", "eventful"):
        run = axiom_run(name)
        steps, state = run["steps"], run["state"]
        wrong_cells = sum(row["wrong_cells"] for row in steps)
        wrong_boxes = sum(row["wrong_boxes"] for row in steps)
        revived = sum(row["revived"] for row in steps)
        print(f"{name}: cells whose own boxes do not sum to their quanta {wrong_cells}, boxes of the wrong cell kind "
              f"or left in a dead id {wrong_boxes}, ids used again {revived} (summed over all steps); one block of "
              f"{BLOCK}x{BLOCK}x{BLOCK} boxes per id: {tuple(state.quanta.shape[1:])}")
        assert wrong_cells == 0 and wrong_boxes == 0 and revived == 0
        assert tuple(state.quanta.shape[1:]) == (BLOCK, BLOCK, BLOCK) and state.quanta.shape[0] == state.capacity


@needs_gpu
def test_making_room_for_new_cells_moves_no_box():
    state, settings = new_universe(4, unevenness=0.0, spare=0.0), Settings(crossing_rate=0.0)
    room = state.capacity
    torn = [0, 2, 8, 10]
    for volume in torn:
        state.fill(volume, int(0.4 * 648000))
    blocks = {cell: state.block(cell) for cell in range(64) if cell not in torn}
    start = totals(state)
    step(state, Settings(crossing_rate=0.0, fluid_pressure=0.0, pull_strength=0.0, clock_packing=0.0, step_time=1e-9))
    after, found = totals(state), pattern(state)
    same = sum(int(np.array_equal(state.block(cell)[0], quanta)) for cell, (quanta, _) in blocks.items())
    print(f"4 places a side with no spare ids: room for {room} ids grew to {state.capacity} when {state.tears} volumes "
          f"tore; cells whose boxes are untouched {same} of {len(blocks)}; amount {start['amount']} -> "
          f"{after['amount']}; pattern {found}")
    assert state.capacity > room and state.tears == 4 and state.count == 72
    assert same == len(blocks) and after["amount"] == start["amount"]
    assert after["wrong_boxes"] == 0 and after["wrong_cells"] == 0
    assert found["web_pieces"] == 1 and not any(found[name] for name in _BREAKS)
    for _ in range(20):
        step(state, settings)
    assert totals(state)["wrong_cells"] == 0 and pattern(state)["web_pieces"] == 1
