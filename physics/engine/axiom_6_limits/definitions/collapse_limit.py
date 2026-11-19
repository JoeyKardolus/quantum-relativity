"""Collapse limit: where the pull is so deep that nothing could climb out of it.

The law, two lines. The top speed is one speed for the whole universe, `TOP_SPEED` calm
lengths a turn of a calm swirl. A volume is past the limit where twice the pull at it is at
least that speed squared: what is thrown up at the top speed cannot climb out.

The same limit is used twice, with no second rule: a joint whose two volumes are both past
it has nothing left to hold apart and collapses (`collapse.py`), and out of a volume past
it nothing leaves, by the leak or by the crossing. A thin joint and a fat one fall at the
same pull. The pull is counted in calm volumes' pull, as everywhere.

One number has to be chosen, because the engine has no one speed to read it off: the pull
is handed on a cell a step, which is 25 calm lengths a turn between calm cells, some thirty
times that across a fresh joint, and other at other boxes across. Chosen before any run:
110. The least that lets a universe begin is 100.2: all its energy in the first volume,
wrapped in a full grid, makes a pull of 3.03 times that energy there (the pull law summed
by hand: twice 1.516), and a first volume past the limit would never leak. 110 is the next
round number with room above that; the fit may search it.
"""
TOP_SPEED = 110.0
DEEP = TOP_SPEED ** 2 / 2.0  # the pull past which nothing climbs out: 6,050 calm volumes' pull


def past_limit(pull):
    """Whether the pull `pull` is past the limit: too deep to climb out of at the top speed."""
    return 2.0 * pull >= TOP_SPEED ** 2
