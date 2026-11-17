"""Expansion scene: does the universe grow, and does its growth slow, then speed up?

Contract: `expansion(cells_per_side=16, seed=1, unevenness=0.1, steps=200, every=10,
settings=None, device="cuda")` builds physics.engine.new_universe(cells_per_side, seed,
unevenness), runs `steps` steps with settings (default Settings(), every rule on),
measuring at the start and every `every` steps (sampled_run.py). The size of the
universe is the sum of its cells' lengths (bible: it wraps, there is no edge and no
empty space; it grows or shrinks purely from where the energy sits). Returns a dict
(numbers and lists of floats, no verdict):
  cells_per_side, seed, unevenness, steps, settings (dict)
  time             sample times
  size             stats "size" per sample; size_growth over the start's
  volume_size, joint_size   its two parts per sample
  rate_time        midpoints between samples
  size_rate        d ln(size)/dt there (growth_rates.py)
  size_rate_trend_early, size_rate_trend_late   slope of size_rate against time over
                   the first and the second half of rate_time (least squares); negative
                   then positive = slowed, then sped up. NaN with < 2 points.
  lumps, collapses, tears, made, taken, books_gap   stats at the end (taken: quanta
                   collapses took, joints and volumes together; books_gap 0 is exact)
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Settings, new_universe, stats
from physics.scenes.growth_rates import growth_rates
from physics.scenes.sampled_run import sampled_run

_KEPT = ("size", "size_growth", "volume_size", "joint_size")


def expansion(cells_per_side: int = 16, seed: int = 1, unevenness: float = 0.1, steps: int = 200, every: int = 10,
              settings: Settings | None = None, device: str = "cuda") -> dict:
    """The universe's size over time, its growth rate and whether the rate turns."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed, unevenness, device=device)
    samples = sampled_run(universe, settings, steps, every,
                          lambda sampled: {name: stats(sampled, settings)[name] for name in _KEPT})
    times = [time for time, _ in samples]
    rate_time, size_rate = growth_rates(times, [measured["size"] for _, measured in samples])
    half = rate_time.size // 2
    final = stats(universe, settings)
    return {"cells_per_side": cells_per_side, "seed": seed, "unevenness": unevenness, "steps": steps,
            "settings": asdict(settings), "time": times,
            **{name: [measured[name] for _, measured in samples] for name in _KEPT},
            "rate_time": rate_time.tolist(), "size_rate": size_rate.tolist(),
            "size_rate_trend_early": _trend(rate_time[:half], size_rate[:half]),
            "size_rate_trend_late": _trend(rate_time[half:], size_rate[half:]),
            "lumps": final["lumps"], "collapses": final["collapses"], "tears": final["tears"],
            "made": final["made"], "taken": final["taken_joint"] + final["taken_volumes"],
            "books_gap": final["books_gap"]}


def _trend(times: np.ndarray, rates: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    slope, _ = np.polyfit(times, rates, 1)
    return float(slope)
