"""Timing: how many steps a throughput timing takes, and which plans are refused.

`STEPS` timed steps follow `WARMUP` untimed ones (kernel compiles and the first step's
set-up are not the engine's speed). `check_timing(cells, steps, warmup)` raises
ValueError for steps below 1, warmup below 0 or a cell count below 1.
"""
STEPS = 20
WARMUP = 1


def check_timing(cells: int, steps: int, warmup: int) -> None:
    """Refuse a timing that cannot give a rate."""
    if steps < 1 or warmup < 0 or cells < 1:
        raise ValueError(f"need steps >= 1, warmup >= 0 and cells >= 1, got {steps}, {warmup}, {cells}")
