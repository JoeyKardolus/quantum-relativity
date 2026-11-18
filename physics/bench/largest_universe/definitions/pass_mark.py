"""Pass mark: the rate a universe must run at to count, written before any run.

The SIZE goal (issue #3) is the largest universe at one step a second: `MIN_RATE`.
`meets_rate(steps_per_second, min_rate=MIN_RATE)` is the mark itself: at or above passes.
`check_rate(min_rate)` raises ValueError for a mark that is not above 0. A size probe
times `STEPS` steps after `WARMUP` untimed ones.
"""
MIN_RATE = 1.0
STEPS = 3
WARMUP = 1


def meets_rate(steps_per_second: float, min_rate: float = MIN_RATE) -> bool:
    """True when a measured rate is at or above the mark."""
    return steps_per_second >= min_rate


def check_rate(min_rate: float) -> None:
    """Refuse a mark every universe would pass."""
    if min_rate <= 0:
        raise ValueError(f"the rate to meet must be above 0, got {min_rate}")
