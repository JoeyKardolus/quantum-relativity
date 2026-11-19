"""Pull and push: the two forces on a joint, from the pull at its two volumes and its own quanta.

Quanta are counted in calm volumes' quanta: a calm volume holds 1 and is 1 long.
Apart: an energy volume is round, a ball as wide as its size, and a joint fills the room
between its two. The cube on the distance between two volumes' middles holds one ball (half
of each of the two) and three joints' room: a joint's room, its size to the power 3, sets
how far apart its two volumes stand. More quanta, less room, closer together.

Pull across: `PULL` times the pull at the one volume (`pull.py`: its own energy and what
all the others hand on) times the pull at the other, over that distance squared. Push back: how packed the joint is, its quanta over its room: quanta to the
power `STEEP`. A joint is squeezed while the pull is the bigger, and gives by `GIVE`.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import slot_at

# The push's one setting, chosen before any run: 4, which is 1 for the quanta and 3 for the
# room (a size in three directions). It is what the size law gives; nothing was fitted.
STEEP = 4.0
# The squeeze's one setting, restated before any run of the spreading pull (2026-11-19): two
# calm volumes alone hold their joint a quarter of their own size thin where it is thinnest.
# They are 1.25 apart, the joint holds 1.28, and each feels a pull of 1.2 (its own 1 and
# what comes back through the joint): 1.28 ** 4 * 1.25 ** 2 / 1.2 ** 2.
PULL = 2.92
# How fast a joint gives way, chosen before any run: to a wholly unanswered pull, a quarter
# in one turn of a calm swirl.
GIVE = 0.25
BALL = 0.5235987755982988  # the room of a ball 1 wide: pi over 6


@wp.func
def apart_of(left: wp.float64, right: wp.float64, joint: wp.float64) -> wp.float64:
    """How far apart the middles of two volumes stand that hold a joint of `joint` quanta."""
    cube = wp.float64(-3.0)
    balls = wp.float64(0.5 * BALL) * (wp.pow(left, cube) + wp.pow(right, cube))
    return wp.cbrt(wp.float64(3.0) * wp.pow(joint, cube) + balls)


@wp.func
def pull_across(before: wp.float64, after: wp.float64, apart: wp.float64) -> wp.float64:
    """The pull across a joint whose volumes feel the pulls `before` and `after`, `apart` apart."""
    return wp.float64(PULL) * before * after / (apart * apart)


@wp.func
def push_back(joint: wp.float64) -> wp.float64:
    """The push of a joint that holds `joint` quanta (counted above 0)."""
    return wp.pow(joint, wp.float64(STEEP))


@wp.func
def given(squeeze: wp.vec4d, a_turn: wp.float64) -> wp.float64:
    """What a joint gives in a step that is `a_turn` of a swirl's turn; 0 when not squeezed.

    `GIVE` times the share of the pull (`squeeze[0]`) that the push (`squeeze[1]`) leaves.
    """
    if squeeze[0] <= squeeze[1]:
        return wp.float64(0.0)
    return wp.float64(GIVE) * a_turn * (squeeze[0] - squeeze[1]) / squeeze[0]


@wp.func
def pressed(box_quanta: wp.array2d(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
            tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
            ends: wp.vec2i, faces: wp.vec2i) -> wp.float64:
    """How hard the two volumes `ends` press on one row of their joint: 1 is their mean.

    The plus quanta of the two boxes `faces` that face each other across the row, each over
    its volume's mean box.
    """
    press = wp.float64(0.0)
    for end in range(2):
        press += (wp.float64(box_quanta[slot_at(tile_home, tile_slots, ends[end]), faces[end]])
                  / wp.float64(cell_quanta[ends[end]]))
    return wp.float64(0.5 * float(box_quanta.shape[1])) * press
