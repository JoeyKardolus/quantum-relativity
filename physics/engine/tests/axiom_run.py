"""Axiom run (test helper): the default universe, seed 1, 300 steps, measured at every step and every 10th, run once per session.

Contract: `axiom_run()` builds `new_universe(seed=1)` with `Settings()` (nothing changed),
steps it STEPS times and returns a dict, cached so every axiom test shares the one run:
  start   totals() at step 0 (amount, plus, minus, boxes)
  kinds   per cell id at the start: 1 energy volume, -1 joint, 0 no cell (state.kind)
  steps   one dict per step, after it: step, amount, plus, minus, boxes; tallies (quanta
          crossed, collapsed boxes, owner changes, collapsed quanta, cumulative); splits, torn;
          touching (26-neighbour box pairs of two different volumes) and open_boxes
          (plus boxes beside empty space or a wall, axiom_contacts.py); flips (boxes whose
          sign differs from the step before: minus->plus marked collapse, minus->plus
          that changed owner (emptied and refilled by another cell, rehoming), minus->plus
          with neither mark; then the same three for plus->minus with the split mark);
          cell_amount (int64 per cell id, its quanta ignoring sign after the step);
          cell_size (float per cell id, the size its rule gives it, state.size: a
          joint's swinging size); crossed (int64 per cell id, the quanta that crossed
          each joint this step from between[0] to between[1], state.crossed) and
          between (the joints' volumes that step); volumes (energy volumes alive);
          collapses (joints collapsed so far)
  cell_amount0  cell_amount at step 0
  checks  at step 0 and every CHECK_EVERY steps: step; cells (cells() of the boxes:
          kind, boxes, quanta, clock, place, between, volumes); web_pieces (face-connected
          pieces of all minus boxes, joined round the edges when the universe wraps);
          pull_reach (largest Chebyshev distance of a box with
          nonzero pull beyond the bounding box of the plus boxes, in boxes);
          plus_spread (how far the plus boxes' bounding box reaches beyond the start's);
          pull_cover (share of occupied boxes with nonzero pull); pressure, size,
          amount per cell id (axiom_pressure.py)
The default universe wraps round at its edges (new_universe, margin 0).
Nothing is tuned; every number is the engine's.
"""
from __future__ import annotations

import functools

import numpy as np
import warp as wp
from physics.engine import Settings, cells, group_label, host_boxes, new_universe, step, totals
from physics.engine.tests.axiom_contacts import axiom_contacts
from physics.engine.tests.axiom_pressure import axiom_pressure

STEPS = 300
CHECK_EVERY = 10


@wp.kernel
def _flips(before: wp.array3d(dtype=wp.int32), after: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8),
           sums: wp.array(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    old = before[ti, tj, tk]
    new = after[ti, tj, tk]
    mark = marks[ti, tj, tk]
    kind = int(-1)
    if old < 0 and new > 0:
        kind = 0
    elif old > 0 and new < 0:
        kind = 3
    if kind < 0:
        return
    rule = wp.uint8(1)
    if kind == 3:
        rule = wp.uint8(8)
    if (mark & rule) == rule:
        wp.atomic_add(sums, kind, wp.int64(1))
    elif (mark & wp.uint8(2)) == wp.uint8(2):
        wp.atomic_add(sums, kind + 1, wp.int64(1))
    else:
        wp.atomic_add(sums, kind + 2, wp.int64(1))


@wp.kernel
def _cell_amount(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
                 sums: wp.array(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held != 0:
        wp.atomic_add(sums, owner[ti, tj, tk], wp.int64(wp.abs(held)))


@functools.lru_cache(maxsize=1)
def axiom_run() -> dict:
    """The default universe's 300 steps, measured; computed once per test session."""
    settings = Settings()
    state = new_universe(seed=1)
    start = totals(state)
    start_plus = _plus_box(host_boxes(state)["quanta"])
    run = {"settings": settings, "start": start, "kinds": state.kind[0].copy(),
           "cell_amount0": _amounts(state), "steps": [], "checks": [_check(state, settings, start_plus)]}
    flips = wp.zeros(6, dtype=wp.int64, device=state.quanta.device)
    for _ in range(STEPS):
        before = wp.clone(state.quanta)
        step(state, settings)
        flips.zero_()
        wp.launch(_flips, dim=state.shape, device=state.quanta.device,
                  inputs=[before, state.quanta, state.marks, flips])
        now = totals(state)
        touching, open_boxes = axiom_contacts(state)
        run["steps"].append({
            "step": state.step, "amount": now["amount"], "plus": now["plus"], "minus": now["minus"],
            "boxes": now["boxes"], "tallies": [int(value) for value in state.tallies.numpy()[0]],
            "splits": int(state.splits[0]), "torn": int(state.torn[0]), "touching": touching,
            "open_boxes": open_boxes, "flips": [int(value) for value in flips.numpy()],
            "cell_amount": _amounts(state), "cell_size": state.size[0].copy(),
            "crossed": state.crossed[0].copy(), "between": state.between[0].copy(),
            "volumes": int((state.kind[0] == 1).sum()), "collapses": int(state.collapses[0])})
        if state.step % CHECK_EVERY == 0:
            run["checks"].append(_check(state, settings, start_plus))
    return run


def _amounts(state) -> np.ndarray:
    """Every cell's quanta ignoring sign, per cell id (universe 0)."""
    sums = wp.zeros(state.capacity, dtype=wp.int64, device=state.quanta.device)
    wp.launch(_cell_amount, dim=state.shape, device=state.quanta.device, inputs=[state.quanta, state.owner, sums])
    return sums.numpy()


def _plus_box(quanta: np.ndarray) -> np.ndarray:
    """The plus boxes' bounding box, (2, 3): low and high index per axis."""
    places = np.argwhere(quanta > 0)
    return np.stack([places.min(axis=0), places.max(axis=0)])


def _beyond(places: np.ndarray, bounds: np.ndarray) -> int:
    """Largest Chebyshev distance of `places` outside `bounds` (0 inside)."""
    if places.size == 0:
        return 0
    outside = np.maximum(np.maximum(bounds[0] - places, places - bounds[1]), 0)
    return int(outside.max())


def _check(state, settings: Settings, start_plus: np.ndarray) -> dict:
    """The heavy measures at one step: cells, web, pull reach and cover, pressures."""
    boxes = host_boxes(state)
    quanta = boxes["quanta"]
    plus_bounds = _plus_box(quanta)
    pulled = np.argwhere(boxes["pull"] != 0)
    pressure, size, amount = axiom_pressure(state, settings)
    return {"step": state.step, "cells": cells(boxes),
            "web_pieces": group_label(quanta < 0, state.wrap)[1],
            "pull_reach": _beyond(pulled, plus_bounds),
            "plus_spread": int(max((start_plus[0] - plus_bounds[0]).max(), (plus_bounds[1] - start_plus[1]).max(), 0)),
            "pull_cover": float((boxes["pull"][quanta != 0] != 0).mean()),
            "pressure": pressure, "size": size, "amount": amount}
