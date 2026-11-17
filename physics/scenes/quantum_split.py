"""Quantum split scene (staged): where does a split energy turn up whole, how often, and can it carry a message?

Contract: `quantum_split(cells_per_side=32, draws=256, packet=1000, seed=1000,
settings=None, device="cuda")` stages axiom 7 on the default universe (new_universe(
cells_per_side, seed), every rule on): `draws` squares of four energy volumes S, A, R1,
R2 that share no volume, a packet of `packet` of S's quanta named in each
(physics.engine quantum_packet). The engine's own crossing takes all direction off what
crosses, so the scene places the splitter (S to A), the two routes (S to R1, A to R2)
and the joining (R1 to R2) by quantum_hand, and a detector by quantum_react. Each
experiment starts from a fresh universe (seed, seed + 1, ...). Returns a dict (shares of
the draws, no verdict):
  cells_per_side, draws, packet, seed, settings (dict)
  half            50/50 split, detector in A: share found whole in S (known 0.5)
  seventy         70/30 split: share found whole in the 70% cell, a list of two, when
                  the 70% half reacts and when the 30% half does (known 0.7 both)
  joined_in_step  two routes brought back together in step: share found whole in exit
                  R2 (known 1); joined_out_of_step the opposite turn (known 0)
  far_with, far_without   70/30 split: share in which the far, 30% side finds the energy
                  with a reaction on the near side first, and without (known 0.3 both:
                  the same, so no message)
Filter-angle (Bell) tests on pairs of whirls are not built. Raises ValueError when the
lattice has no room for `draws` squares.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from physics.engine import Settings, State, new_universe, quantum_hand, quantum_packet, quantum_react, quantum_store, step

_HALF = math.pi / 2.0
_THIRTY = 2.0 * math.asin(math.sqrt(0.3))


def quantum_split(cells_per_side: int = 32, draws: int = 256, packet: int = 1000, seed: int = 1000,
                  settings: Settings | None = None, device: str = "cuda") -> dict:
    """The share of the draws in which a split energy settles on each side, for five staged experiments."""
    settings = settings or Settings()
    squares = _squares(cells_per_side, draws)
    seeds = iter(range(seed, seed + 8))

    def bench() -> State:
        state = new_universe(cells_per_side, seed=next(seeds), device=device)
        for square in squares:
            quantum_packet(state, square[0], packet)
        return state

    state = bench()
    _hand(state, squares, 0, 1, _HALF)
    _run(state, settings, 2)
    half = _landed(_react(state, squares, 1), squares, 0)
    seventy = []
    for reacting in (0, 1):
        state = bench()
        _hand(state, squares, 0, 1, _THIRTY)
        _run(state, settings, 2)
        seventy.append(_landed(_react(state, squares, reacting), squares, 0))
    joined = []
    for turn in (_HALF, -_HALF):
        state = bench()
        _hand(state, squares, 0, 1, _HALF)
        _run(state, settings, 1)
        _hand(state, squares, 0, 2, math.pi)
        _hand(state, squares, 1, 3, math.pi)
        _run(state, settings, 3)
        _hand(state, squares, 2, 3, turn)
        joined.append(_landed(_react(state, squares, 2) + _react(state, squares, 3), squares, 3))
    far = []
    for near_reacts in (True, False):
        state = bench()
        _hand(state, squares, 0, 1, _THIRTY)
        _run(state, settings, 1)
        if near_reacts:
            _react(state, squares, 0)
        _run(state, settings, 2)
        _react(state, squares, 1)
        far.append(_landed(quantum_store(state)["ended"], squares, 1))
    return {"cells_per_side": cells_per_side, "draws": draws, "packet": packet, "seed": seed,
            "settings": asdict(settings), "half": half, "seventy": seventy, "joined_in_step": joined[0],
            "joined_out_of_step": joined[1], "far_with": far[0], "far_without": far[1]}


def _squares(side: int, count: int) -> list:
    """`count` squares (S, A, R1, R2) of energy volumes sharing no volume: A beside S along the first axis, R along the second."""
    found = [tuple((one * side + two) * side + third for one, two in
                   ((first, second), (first + 2, second), (first, second + 2), (first + 2, second + 2)))
             for first in range(0, side - 3, 4) for second in range(0, side - 3, 4) for third in range(0, side, 2)]
    if len(found) < count:
        raise ValueError(f"a {side}-place lattice has room for {len(found)} squares, not {count}")
    return found[:count]


def _run(state: State, settings: Settings, steps: int) -> None:
    for _ in range(steps):
        step(state, settings)


def _hand(state: State, squares: list, giver: int, taker: int, turn: float) -> None:
    """The same handover in every square, between its corners number `giver` and `taker`."""
    for square in squares:
        quantum_hand(state, square[giver], square[taker], turn)


def _react(state: State, squares: list, corner: int) -> list:
    return [event for square in squares for event in quantum_react(state, square[corner])]


def _landed(events: list, squares: list, corner: int) -> float:
    """The share of the events that settled in corner number `corner` of their square (0 with no event)."""
    homes = {square[corner] for square in squares}
    return sum(event["cell"] in homes for event in events) / max(len(events), 1)
