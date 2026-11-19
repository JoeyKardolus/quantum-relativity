"""Pull: energy attracts energy, pair by pair, after a delay.

The law, three lines. Every volume is pulled toward every other volume by that one's plus
energy (in calm volumes: more packed, more in one place, stronger) over the square of the
distance between them. The distance is counted through the cells' sizes: along each of the
three directions the sizes of the cells that lie between the two (`logic/between.py`). The
pull travels at the one top speed (`TOP_SPEED`), so a volume reads the other's energy as it
was the distance over that speed ago, never sooner than a step.

There is no self term: a volume is never in its own sum, so it cannot move itself whatever
its joints look like. The way from one volume to another is the other's way turned round,
so two volumes that hold steady pull each other equally and oppositely.

How deep a volume lies (`cell_pull`, what the squeeze and the collapse limit read) is its
own energy and every other's over its distance, read as late. No number is set: a calm
volume pulls 1 at one calm length, the units the pull was already counted in.

Cost. On the card, a thread a volume. All pairs (`near` 0) is the law with nothing left
out, right for some thousand volumes. For more the far pull is read off groups
(`groups.py`): the work a volume is then the same however many there are.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

# How many steps of the volumes' past are kept. Not physics, a bound on memory, chosen before
# any run: 128 steps is 563 calm lengths of travel at 6 across, nine fresh joints in a row.
# A pull from further off is read as the oldest kept.
REACH = 128


def steps_late(apart: np.ndarray, a_turn: float) -> np.ndarray:
    """How many steps ago a pull from `apart` calm lengths off set out; a step is `a_turn` of a calm turn."""
    return np.clip(np.ceil(apart / (TOP_SPEED * a_turn)), 1, REACH - 1).astype(int)


@wp.func
def late_of(apart: wp.float64, a_step: wp.float64) -> wp.int32:
    """On the card: `steps_late`, the top speed going `a_step` calm lengths a step."""
    return wp.clamp(wp.int32(wp.ceil(apart / a_step)), 1, REACH - 1)
