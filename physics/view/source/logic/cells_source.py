"""Cells source: the engine (physics.engine) behind the viewer's source interface.

Contract: `CellsSource(cells_per_side=32, seed=1, overrides=None, device="cuda",
scene=None, quantum=True)` fits `Source`. `scene` names a gravity scene
(`physics.scenes.gravity`: "two_lumps", "coast", "orbit", "lone"): the start is then that
scene's placed lumps instead of the uneven universe. `quantum=False` steps the layers
below axiom 7 only (phase 1 of the fit). Settings are the engine's `Settings(**overrides)`; an unknown name raises
TypeError and a value the engine refuses raises ValueError, both at construction.
  reset()   `new_universe(settings, cells_per_side, seed)` or the scene's start, step 0; takes the start's
            packing marks and size, which every later frame is read against
  step()    one `physics.engine.step`
  steps     steps since the last reset
  view(layer=None)  `grid_pieces`: one layer, or the packed cells
  stats()   `cell_stats`
"""
from __future__ import annotations

from physics.engine import Settings, measure_grid, new_universe, step
from physics.view.colours.logic.start_marks import start_marks
from physics.view.layout.definitions.pieces import Pieces
from physics.view.source.logic.cell_stats import cell_stats
from physics.view.source.logic.grid_pieces import grid_pieces


class CellsSource:
    """A universe of cells stepped under all seven axioms, read as the blocks of its warped grid."""

    def __init__(self, cells_per_side: int = 32, seed: int = 1, overrides: dict | None = None,
                 device: str = "cuda", scene: str | None = None, quantum: bool = True) -> None:
        self.cells_per_side, self.seed, self.device = cells_per_side, seed, device
        self.scene, self.quantum = scene, quantum
        self.settings = Settings(**(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same size, seed and settings."""
        if self.scene is None:
            self.state = new_universe(self.settings, cells_per_side=self.cells_per_side, seed=self.seed,
                                      device=self.device)
        else:
            from physics.scenes.gravity.logic.scene_start import scene_start
            self.state = scene_start(self.scene, self.settings, self.cells_per_side, self.seed, self.device)[0]
        self.marks = start_marks(self.state.host("kind"), self.state.host("packing"))
        self.start_size = measure_grid(self.state)["size"]
        self.steps = 0

    def step(self) -> None:
        """One engine step."""
        step(self.state, self.settings, quantum=self.quantum)
        self.steps += 1

    def view(self, layer: int | None = None) -> Pieces:
        """One layer's cells, or the packed cells, as blocks of the drawn grid."""
        return grid_pieces(self.state, self.marks, layer)

    def stats(self) -> dict:
        """The engine's counts for the title."""
        return cell_stats(self.state, self.start_size)
