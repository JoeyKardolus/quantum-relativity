"""Expansion scene: does the universe grow, and does its growth slow, then speed up?

Contract: `expansion(cells_per_side=16, seed=1, steps=200, every=10, settings=None,
device="cuda", start=None)` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a fit candidate's start amounts in), runs `steps`
steps with every rule on, measuring at the start and every `every` steps (sampled_run).
The size of the universe is the sum of its cells' lengths (measure_grid: it wraps,
there is no edge and no empty space; it grows or shrinks purely from where the energy
sits). Returns a dict (numbers and lists of floats, no verdict):
  cells_per_side, seed, steps, settings (dict)
  time             sample times (the counter's)
  size             per sample; size_growth the same over the start's
  volume_size, joint_size   its two parts per sample
  rate_time        midpoints between samples
  size_rate        d ln(size)/dt there (growth_rates)
  size_rate_trend_early, size_rate_trend_late   slope of size_rate against time over
                   the first and the second half of rate_time (rate_trend); negative
                   then positive = slowed, then sped up. NaN with < 2 points.
  lumps, collapses, tears, made, taken, books_gap   at the end (made: minus quanta
                   squeezed joints made; taken: quanta collapses took, joints and
                   volumes together; books_gap: the largest miss of book_law, 0 is exact)
Raises ValueError for steps < 1 or every < 1.
"""
from __future__ import annotations

from dataclasses import asdict

from physics.engine import Grid, Settings, book_law, measure_grid, new_universe, read_books
from physics.scenes.expansion.definitions.growth_rates import growth_rates
from physics.scenes.expansion.definitions.rate_trend import rate_trend
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.volumes.logic.count_lumps import count_lumps

_KEPT = ("size", "volume_size", "joint_size")


def _sizes(universe: Grid) -> dict:
    """The universe's size and its two parts now."""
    measured = measure_grid(universe)
    return {name: measured[name] for name in _KEPT}


def expansion(cells_per_side: int = 16, seed: int = 1, steps: int = 200, every: int = 10,
              settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """The universe's size over time, its growth rate and whether the rate turns."""
    if steps < 1:
        raise ValueError(f"a growth rate needs at least one step, got {steps}")
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    samples = sampled_run(universe, settings, steps, every, _sizes)
    times = [time for time, _ in samples]
    size = [measured["size"] for _, measured in samples]
    rate_time, size_rate = growth_rates(times, size)
    half = rate_time.size // 2
    books = read_books(universe)
    return {"cells_per_side": cells_per_side, "seed": seed, "steps": steps, "settings": asdict(settings),
            "time": times, **{name: [measured[name] for _, measured in samples] for name in _KEPT},
            "size_growth": [value / size[0] for value in size],
            "rate_time": rate_time.tolist(), "size_rate": size_rate.tolist(),
            "size_rate_trend_early": rate_trend(rate_time[:half], size_rate[:half]),
            "size_rate_trend_late": rate_trend(rate_time[half:], size_rate[half:]),
            "lumps": count_lumps(universe, settings)["lumps"], "collapses": int(universe.collapses),
            "tears": int(universe.tears), "made": books["made_total"],
            "taken": books["taken_joint"] + books["taken_volumes"],
            "books_gap": max(abs(gap) for gap in book_law(universe.start_books, books).values())}
