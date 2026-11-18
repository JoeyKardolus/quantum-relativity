"""Draw uniform: the engine's one source of chance, a hash that keeps no state.

The same seed, step, pass and box always give the same number in [0, 1).
"""
import warp as wp

_UNIT = wp.constant(wp.float64(2.0 ** -53))


@wp.func
def _finish(value: wp.uint64) -> wp.uint64:
    """The splitmix64 finish: every bit of the input changes every bit of the output."""
    mixed = (value ^ (value >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return mixed ^ (mixed >> wp.uint64(31))


@wp.func
def draw_uniform(seed: wp.uint32, step: wp.uint32, pass_id: wp.uint32, box: wp.uint64):
    """The draw in [0, 1) for one box in one pass of one step of one seed."""
    mixed = _finish(_finish(wp.uint64(seed) + wp.uint64(0x9E3779B97F4A7C15)) ^ wp.uint64(step))
    mixed = _finish(_finish(mixed ^ wp.uint64(pass_id)) ^ box)
    return wp.float64(mixed >> wp.uint64(11)) * _UNIT
