"""Spin scene: does a whirl placed inside one cell keep its whole turns while every rule runs?

Contract: `spin(cells_per_side=16, steps=2000, cases=CASES, seed=1, settings=None,
start=None, device="cuda")` builds physics.engine.new_universe(cells_per_side, seed,
**start) (the uneven start) and places, per case (kind "volume" or "joint", turns), a
whirl of that many whole turns of phase round the middle line of one cell's own block of
compute boxes; the cells are spread along the lattice's diagonal so no two neighbour
(at most (cells_per_side // 4) ** 2 cases).
It then runs `steps` steps with settings (default Settings(), every rule on) and reads,
after every step, the whole turns round the rim of each layer of that block
(physics.measure.phase_winding, summed over the layer). A whirl lives inside its cell:
no rule carries phase from one cell's boxes to another's, so nothing ties a whirl to the
joint web and a turn of the whole whirl against its surroundings (the belt trick, spin
one half) cannot be asked of this engine. Returns a dict (no verdict):
  cells_per_side, steps, seed, settings (dict)
  cases    one dict per case: joint (1 in a joint, 0 in an energy volume), placed (turns), cell (id),
           seen        every value read round any layer's rim at any step, sorted
           held_steps  steps at which every layer read exactly the placed turns
           first_off   the first step at which one did not (-1: never)
           end         the turns round each layer's rim at the end
Raises ValueError for a kind that is neither "volume" nor "joint".
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import BLOCK, Settings, new_universe, step
from physics.measure import phase_winding

CASES = (("volume", 1), ("volume", -1), ("joint", 1), ("joint", -1), ("volume", 2))


def spin(cells_per_side: int = 16, steps: int = 2000, cases: tuple = CASES, seed: int = 1,
         settings: Settings | None = None, start: dict | None = None, device: str = "cuda") -> dict:
    """Per placed whirl: the turns read round its cell's layers at every step of a run with all rules on."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed, device=device, **(start or {}))
    side = universe.side
    placed = []
    for number, (kind, turns) in enumerate(cases):
        if kind not in ("volume", "joint"):
            raise ValueError(f"kind must be volume or joint, got {kind}")
        along, round_again = (2 + 4 * number) % side, (2 + 4 * (number // max(side // 4, 1))) % side
        cell = (along * side + along) * side + (round_again + (kind == "joint")) % side
        universe.set_block(cell, universe.block(cell)[0], _whirl(turns))
        placed.append({"joint": int(kind == "joint"), "placed": turns, "cell": cell, "seen": set(), "held_steps": 0, "first_off": -1})
    for taken in range(1, steps + 1):
        step(universe, settings)
        for case in placed:
            turns = _rim_turns(*universe.block(case["cell"]))
            case["seen"] |= set(turns)
            if set(turns) == {case["placed"]}:
                case["held_steps"] += 1
            elif case["first_off"] < 0:
                case["first_off"] = taken
            case["end"] = turns
    return {"cells_per_side": cells_per_side, "steps": steps, "seed": seed, "settings": asdict(settings),
            "cases": [{**case, "seen": sorted(case["seen"])} for case in placed]}


def _whirl(turns: int) -> np.ndarray:
    """`turns` whole turns of phase round the middle line (along the third axis) of a block."""
    spot = np.arange(BLOCK) - (BLOCK - 1) / 2.0
    first, second = np.meshgrid(spot, spot, indexing="ij")
    angle = np.angle(np.exp(1j * turns * np.arctan2(second, first)))
    return np.repeat(angle[:, :, None], BLOCK, axis=2).astype(np.float32)


def _rim_turns(quanta: np.ndarray, phase: np.ndarray) -> list:
    """The whole turns of phase round the rim of each layer of a block: the unit squares inside it, summed."""
    field = np.sqrt(np.abs(quanta).astype(np.float64)) * np.exp(1j * phase.astype(np.float64))
    return [int(turns) for turns in phase_winding(field)[2].sum(axis=(0, 1))]
