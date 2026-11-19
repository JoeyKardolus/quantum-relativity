"""The least whirl is the same share of a cell at any boxes across."""
from physics.engine.axiom_6_limits.definitions.least_whirl import PARTS, least_whirl


def test_the_least_whirl_is_the_same_share_of_a_cell_at_any_boxes():
    assert least_whirl(6) == 648_000 / 60 == 50 * 216 and least_whirl(12) == 8 * least_whirl(6)
    assert PARTS == 7  # a volume and its six sides
