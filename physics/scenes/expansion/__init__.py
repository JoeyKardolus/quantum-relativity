"""Expansion shape scene: does the universe grow, first slowing, then speeding up?

  definitions/  growth_rates, rate_trend, the pass mark
  logic/        expansion (the scene)
"""
from physics.scenes.expansion.definitions.growth_rates import growth_rates
from physics.scenes.expansion.definitions.pass_mark import GROWTH, passes
from physics.scenes.expansion.definitions.rate_trend import rate_trend
from physics.scenes.expansion.logic.expansion import expansion

__all__ = ["GROWTH", "expansion", "growth_rates", "passes", "rate_trend"]
