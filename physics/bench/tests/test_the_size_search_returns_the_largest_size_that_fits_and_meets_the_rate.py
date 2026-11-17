"""The size search bisects to the largest size that both fits and runs at the asked rate.

Host-only fakes: a step that sleeps 0.1 s per size unit, asked for 3 steps a second
(sizes 1 and 2 pass, 4 and up fail); and a build that raises MemoryError past size 5
with instant steps (size 5 is the largest).
"""
import time

from physics.bench import largest_universe


def test_rate_limit_and_memory_limit():
    slow = largest_universe(lambda size: ((lambda: time.sleep(0.1 * size)), size ** 3), [1, 2, 4, 8, 16],
                            min_rate=3.0, steps=1, warmup=0)
    assert slow["size"] == 2 and slow["cells"] == 8 and slow["steps_per_second"] >= 3.0
    assert len(slow["probes"]) <= 3

    def build(size):
        if size > 5:
            raise MemoryError("too big")
        return (lambda: None), size

    fits = largest_universe(build, list(range(1, 11)), steps=1, warmup=0)
    assert fits["size"] == 5
    assert any(not probe["fits"] for probe in fits["probes"])

    never = largest_universe(lambda size: ((lambda: time.sleep(0.2)), size), [1, 2], min_rate=100.0, steps=1,
                             warmup=0)
    assert never["size"] is None
