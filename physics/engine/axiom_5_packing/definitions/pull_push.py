"""Pull and push: the two forces on a joint, each from the cells' quanta alone.

Every quanta here is counted in calm volumes' quanta, so a calm volume holds 1 and is 1 long.

Pull across: two volumes pull on each other with `PULL` times the one's quanta times the
other's, over the square of the distance between them: half the one's size, the joint's
size and half the other's, each size by the size law (1 over its quanta).

Push back: a joint pushes with how packed it is, its quanta over its room. Its room is its
size to the power 3, so the push is its quanta to the power `STEEP`.

A joint is squeezed while the pull is the bigger, and gives way by `GIVE` (see `given`).
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import slot_at

# The push's one setting, chosen before any run: 4, which is 1 for the quanta and 3 for the
# room (a size in three directions). It is what the size law gives; nothing was fitted.
STEEP = 4.0
# The pull's one setting, chosen before any run: two calm volumes hold their joint at half
# their own size (the owner: a joint is narrower than its volumes; the half is assumed).
# There the joint holds 2, pushes 2 ** 4 = 16, and the distance is 1.5: 16 * 1.5 ** 2 = 36.
PULL = 36.0
# How fast a joint gives way, chosen before any run: to a wholly unanswered pull, a quarter
# in one turn of a calm swirl. Slow beside the fluid, which goes round several times while
# the joint shrinks; quick enough to settle in a run of seconds.
GIVE = 0.25


@wp.func
def pull_across(left: wp.float64, right: wp.float64, joint: wp.float64) -> wp.float64:
    """The pull of two volumes on each other across their joint; 0 where one of the three is empty."""
    if left <= wp.float64(0.0) or right <= wp.float64(0.0) or joint <= wp.float64(0.0):
        return wp.float64(0.0)
    apart = wp.float64(0.5) / left + wp.float64(1.0) / joint + wp.float64(0.5) / right
    return wp.float64(PULL) * left * right / (apart * apart)


@wp.func
def push_back(joint: wp.float64) -> wp.float64:
    """The push of a joint that holds `joint` quanta (counted above 0)."""
    return wp.pow(joint, wp.float64(STEEP))


@wp.func
def given(squeeze: wp.vec2d, a_turn: wp.float64) -> wp.float64:
    """What a joint gives in a step that is `a_turn` of a swirl's turn; 0 when not squeezed.

    `GIVE` times the share of the pull (`squeeze[0]`) that the push (`squeeze[1]`) leaves.
    """
    if squeeze[0] <= squeeze[1]:
        return wp.float64(0.0)
    return wp.float64(GIVE) * a_turn * (squeeze[0] - squeeze[1]) / squeeze[0]


@wp.func
def pressed(box_quanta: wp.array2d(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
            tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
            joint: wp.int32, row: wp.int32, across: wp.int32) -> wp.float64:
    """How hard the two volumes' fluid presses on one row of a joint: 1 is their mean.

    The plus quanta of the two boxes that face each other across the row, each over its
    volume's mean box. `row` is the row's first box, which faces the left volume.
    """
    left = box_quanta[slot_at(tile_home, tile_slots, joint - 1), row + (across - 1) * across]
    right = box_quanta[slot_at(tile_home, tile_slots, joint + 1), row]
    return wp.float64(0.5 * float(box_quanta.shape[1])) * (
        wp.float64(left) / wp.float64(cell_quanta[joint - 1])
        + wp.float64(right) / wp.float64(cell_quanta[joint + 1]))
