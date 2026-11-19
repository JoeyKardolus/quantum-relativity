"""Collapse limit: where the pull at a joint is so deep that nothing could climb out of it.

The law, two lines. The top speed is the pull's own: it is handed from a volume to the
joint and on to the next volume, so it goes the joint's distance in two steps. A joint is
past its limit when twice the pull at it is at least that speed squared: what a body
thrown up at the top speed has is no more than the depth takes from it.

Units, the engine's own: the pull in calm volumes' pull, the distance in calm lengths, the
time in turns of a calm swirl. No number is set here; that a calm volume's pull of 1 is a
speed squared of 1 in those units is the one reading the law rests on.
"""


def top_speed(apart, a_turn: float):
    """The top speed where two volumes stand `apart` apart, in calm lengths a turn (a step: `a_turn` of one)."""
    return apart / (2.0 * a_turn)


def past_limit(pull, apart, a_turn: float):
    """Whether a joint with the pull `pull` at it, its volumes `apart` apart, is past its limit."""
    return (apart > 0) & (2.0 * pull >= top_speed(apart, a_turn) ** 2)
