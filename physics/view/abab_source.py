"""ABAB source: physics.abab's State behind the viewer's source interface.

Contract (the source interface; the GPU engine provides the same):
  reset()      back to a fresh start: `new_state(volumes, seed, unevenness)`, step 0
  step()       one engine step, `physics.abab.step(state, settings)`
  steps        steps since the last reset
  positions    (n, 3) float array, where each volume is (`state.position` as it is)
  kind         (n,) int array, KIND_A energy volume or KIND_B joint
  density      (n,) float array, energy density: energy / size
  stats()      dict with volumes (energy volumes plus joints), collapses, splits,
               size_vs_start, lumps, sealed_lumps (`physics.abab.stats`)
Arrays may be numpy or anything with `.numpy()` (a Warp array). Settings are
`Settings.all_on(**overrides)`; an unknown name raises TypeError at construction.
step() raises ValueError when an energy or size stops being finite and positive or
a position stops being finite; that state is not taken.
"""
from __future__ import annotations

import numpy as np

from physics.abab import Settings, new_state, stats, step


class AbabSource:
    """An ABAB state stepped under all rules on, read as positions, kinds and densities."""

    def __init__(self, volumes: int = 2000, seed: int = 1, unevenness: float = 0.5,
                 overrides: dict[str, float] | None = None) -> None:
        self.volumes, self.seed, self.unevenness = volumes, seed, unevenness
        self.settings = Settings.all_on(**(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same volumes, seed and unevenness."""
        self.state = new_state(self.volumes, self.seed, self.unevenness)
        self.steps = 0

    def step(self) -> None:
        """One engine step; a state that broke is refused."""
        with np.errstate(all="ignore"):
            stepped = step(self.state, self.settings)
        for name in ("energy", "size"):
            values = np.asarray(getattr(stepped, name), dtype=float)
            if not (np.all(np.isfinite(values)) and np.all(values > 0)):
                raise ValueError(f"step {self.steps + 1}: a {name} stopped being finite and positive")
        if not np.all(np.isfinite(stepped.position)):
            raise ValueError(f"step {self.steps + 1}: a position stopped being finite")
        self.state = stepped
        self.steps += 1

    @property
    def positions(self) -> np.ndarray:
        """Where each volume is."""
        return self.state.position

    @property
    def kind(self) -> np.ndarray:
        """Energy volume or joint, per volume."""
        return self.state.kind

    @property
    def density(self) -> np.ndarray:
        """Energy over size, per volume."""
        return self.state.energy / self.state.size

    def stats(self) -> dict:
        """Volumes, collapses, splits, size against the start, lumps, sealed lumps."""
        counts = stats(self.state, self.settings)
        return {"volumes": counts["energy_volumes"] + counts["joints"], **counts}
