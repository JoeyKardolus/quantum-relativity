"""The pull adds to a volume's flow toward the deeper side, by the others' pull alone; an even pull adds nothing."""
from physics.engine import calm_cells, line_cells, three_cells
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_5_packing.logic.fall import fall
from physics.engine.axiom_5_packing.logic.spread import spread
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

LINE = [5 * steps for steps in range(5)]  # volume, joint, volume, joint, volume along direction 1


def _fallen(pulls: tuple, steps: int = 1):
    """Five calm cells in a line, the three volumes' pulls held as given: the volumes' flow along the line."""
    store = calm_cells(125, 5)
    store.side = 5
    store.cell_quanta.zero_()
    for cell, halves in zip(LINE, (2, -2, 2, -2, 2)):
        put(store.cell_quanta, cell, halves * 324_000)
        take_own(store, cell)
    for _ in range(steps):
        for cell, pull in zip(LINE, (pulls[0], 0, pulls[1], 0, pulls[2])):
            put(store.cell_pull, cell, pull)
        fall(store)
    return store.cell_flow.numpy()[LINE[::2]]


def test_an_even_pull_adds_no_flow_and_a_slope_adds_it_toward_the_deeper_side():
    assert (_fallen((0, 0, 0)) == 0).all() and (_fallen((2, 2, 2))[1] == 0).all()
    assert _fallen((1, 2, 3))[1, 1] > 0 > _fallen((3, 2, 1))[1, 1]  # the middle volume's fluid, up the slope
    assert (_fallen((1, 2, 3))[:, [0, 2]] == 0).all()  # and only along the line
    # the slope of 2 over the 4 calm lengths between its neighbours, a 25th of a turn: 648,000 * 0.5 / 25
    assert abs(int(_fallen((1, 2, 3))[1, 1]) - 12_960) < 60


def test_a_volume_is_pulled_by_what_the_other_holds_not_by_its_own_pull_handed_back():
    store = three_cells()  # A holds 1.5 calm volumes, B a half
    a, _joint, b = line_cells()
    for _ in range(50):
        spread(store)  # the pull settled, nothing else moving
    fall(store)
    flow = store.cell_flow.numpy()[[a, b], 1] / [972_000, 324_000]  # the speed each gained
    assert flow[0] > 0 > flow[1]  # each toward the other
    assert 2.5 < -flow[1] / flow[0] < 3.5  # B, pulled by 1.5, gains three times what A gains, pulled by a half


def test_nothing_is_set_going_faster_than_the_top_speed():
    flow = _fallen((0, 0, 1e7), steps=3)
    assert flow[1, 1] == int(648_000 * TOP_SPEED)
