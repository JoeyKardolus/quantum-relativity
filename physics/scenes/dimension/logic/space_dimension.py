"""Space dimension scene: the ball and the spectral dimension of the volume graph, beside a plain grid's.

Contract: `space_dimension(cells_per_side=16, steps=20, sources=8, seed=1, settings=None,
device="cuda", start=None)` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a candidate's start amounts in), measures it
(graph_dimension), takes `steps` steps with every rule on and measures it again. The
same measure run on the reference grid (reference_grid: a plain wrapping grid of as many
nodes a side as the start has energy volumes a side, 26 neighbours each) gives the
number the estimator returns for true dimension 3 at this size and shape, since a small
graph pulls both estimates off 3. Returns a dict (no verdict):
  cells_per_side, steps, seed, settings (dict), known_dimension (3)
  collapses, tears        events during the run: with none, end equals start
  reference, start, end   each what measure_dimension returns
"""
from __future__ import annotations

from dataclasses import asdict

from physics.engine import Settings, new_universe, step
from physics.measure import measure_dimension
from physics.scenes.dimension.definitions.known import KNOWN_DIMENSION
from physics.scenes.dimension.definitions.reference_grid import reference_grid
from physics.scenes.dimension.logic.graph_dimension import graph_dimension
from physics.scenes.run.definitions.start_settings import start_settings


def space_dimension(cells_per_side: int = 16, steps: int = 20, sources: int = 8, seed: int = 1,
                    settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Ball and spectral dimension of the volume graph at the start and after the steps, beside the plain grid's."""
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    before = graph_dimension(universe, sources)
    for _ in range(steps):
        step(universe, settings)
    after = graph_dimension(universe, sources)
    return {"cells_per_side": cells_per_side, "steps": steps, "seed": seed, "settings": asdict(settings),
            "known_dimension": KNOWN_DIMENSION, "collapses": int(universe.collapses), "tears": int(universe.tears),
            "reference": measure_dimension(reference_grid(cells_per_side // 2), sources), "start": before, "end": after}
