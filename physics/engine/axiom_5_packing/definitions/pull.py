"""Pull: one number a cell, handed on from cell to cell, and how it turns crossing energy.

The law, two lines. A volume's pull is its own plus energy, counted in calm volumes (more
packed, stronger), and a sixth of the pull of the joint at each of its six sides; a side
with no joint gives none. A joint's pull is the mean of its two volumes': it carries the
pull and adds none. Both read the pull as it was a step ago, so it goes one neighbour a
step and no faster; it fades with distance because every volume hands it on in six parts.
The law has no setting.

What it does to energy: where the pull differs across a joint by `differs` (in calm volumes'
own pull), the fluid of the two facing boxes is turned, the deeper side's phase back and the
other's on, by `BEND` times `differs / (1 + differs)`: in step with the difference while it
is small, and never more than `BEND`. Fluid flows from the phase that is ahead, so energy
crosses toward the deeper pull; the turn stays in the fluid as its motion, less what
friction takes. Stated a cell, not a box: `cross.py` hands `a_turn` over `layer_part`.
"""
import warp as wp

# The bend's one setting, chosen before any run: a quarter, as `GIVE`. The steepest pull
# turns the crossing fluid a quarter turn in one turn of a calm swirl; the calm volume's own
# pull is the mark where it bends half of that, as the leak's law has it: no new size.
BEND = 0.25
SIDES = 6.0  # a volume's sides: no number of its own


@wp.func
def differing(low: wp.float32, high: wp.float32) -> wp.float64:
    """How far the pull across a joint bends, of the most it can: above 0, deeper at the volume after."""
    return wp.float64((high - low) / (1.0 + wp.abs(high - low)))


@wp.func
def bent(low_phase: wp.float32, high_phase: wp.float32, share: wp.float64,
         a_turn: wp.float64) -> wp.vec2f:
    """The two facing phases turned by the pull in a step that is `a_turn` of a swirl's turn."""
    half = float(wp.float64(0.5 * BEND * wp.tau) * share * a_turn)
    turn = float(wp.tau)
    low, high = low_phase + half, high_phase - half
    return wp.vec2f(low - turn * wp.floor((low + float(wp.pi)) / turn),
                    high - turn * wp.floor((high + float(wp.pi)) / turn))
