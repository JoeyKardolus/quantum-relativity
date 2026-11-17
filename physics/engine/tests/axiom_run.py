"""Axiom run (test helper): two real runs checked at every step, each run once per session.

Contract: `axiom_run(name)` returns a dict for one of two runs, cached:
  "default"   new_universe() (32 places a side, seed 1) with Settings(), 1,000 steps
  "eventful"  new_universe(16, unevenness=0.3) with Settings(pull_strength=1e-6), ten
              times the default pull, 60 steps: a start that opens with tears and runs
              into a chain of collapses, hundreds of each
with keys:
  start    totals() at step 0; start_kind the kinds at step 0
  steps    one dict per step, measured right after it (every count is over all cells):
    step, amount, plus, minus, made, taken_joint, taken_volumes, torn, crossed (totals.py)
    wrong_boxes, wrong_cells   boxes that break their cell, cells whose boxes do not
                    sum to their quanta
    size_wrong      living cells whose length is not exactly float32(constant / quanta)
    kind_wrong      ids whose kind on the GPU is not the host's
    events          tears and collapses this step (state.version's rise)
    relinked        1 when any relation array on the GPU changed this step
    revived         ids that were no cell before the step and are one after, below the
                    old count (an id used again)
    joint_fell      joints living before and after whose quanta fell
    joint_rose      those whose quanta rose; joint_rose_loose those that rose with a
                    squeeze of 1 or less this step; joint_gain_wrong those whose rise is
                    not their gain
    joint_loss      quanta any joint lost by crossing (state.loss over joints)
    volume_wrong    volumes living before and after, in a step without events, whose
                    quanta did not change by exactly gain - loss
    gained, lost, crossed_step   quanta volumes gained, lost, and that crossed joints
    crossed_at_volumes  state.crossed summed over energy volumes
    clock_wrong     living cells whose clock did not move on by step_time * the rate
                    the step began with (to 1e-4, the float32 clock's rounding)
    still           of 8 sample cells, those whose boxes are the same as a step before
    size, volumes, joints, collapses, merges, tears
    quanta_max_is_shortest  1 when the volume with the most quanta is the shortest
  patterns  [(step, pattern())] at step 0, after every step with events, every 100 steps
  final    stats() at the end; settings; state the state itself (do not step it)
Nothing is tuned; every number is the engine's.
"""
from __future__ import annotations

import functools
import hashlib

import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, Settings, new_universe, pattern, stats, step, totals

RUNS = {"default": ({}, {}, 1000),
        "eventful": ({"cells_per_side": 16, "unevenness": 0.3}, {"pull_strength": 1e-6}, 60)}
_RELATIONS = ("kind", "adj_at", "adj_n", "way_at", "ways_n", "adj_pool", "way_pool")


@functools.lru_cache(maxsize=None)
def axiom_run(name: str = "default") -> dict:
    """Run one of the two universes, measuring every step."""
    made, rules, count = RUNS[name]
    state, settings = new_universe(**made), Settings(**rules)
    run = {"start": totals(state), "start_kind": state.kind_h[:state.count].copy(), "steps": [],
           "patterns": [(0, pattern(state))], "settings": settings, "state": state}
    samples = np.linspace(0, state.lattice - 1, 8).astype(int)
    for _ in range(count):
        before = _before(state, samples)
        step(state, settings)
        run["steps"].append(_after(state, settings, before, samples))
        if run["steps"][-1]["events"] or state.step % 100 == 0:
            run["patterns"].append((state.step, pattern(state)))
    run["final"] = stats(state)
    return run


def _links(state) -> str:
    """One hash of every relation array on the GPU."""
    digest = hashlib.sha1()
    for name in _RELATIONS:
        digest.update(getattr(state, name).numpy().tobytes())
    return digest.hexdigest()


def _before(state, samples) -> dict:
    return {"held": state.host("cell_quanta").copy(), "kind": state.kind_h[:state.count].copy(),
            "clock": state.host("clock").copy(), "rate": state.host("rate").copy(), "version": state.version, "links": _links(state),
            "count": state.count, "blocks": [state.block(cell)[0] for cell in samples]}


def _after(state, settings, before, samples) -> dict:
    found = totals(state)
    old = before["count"]
    kind_now = state.kind_h[:state.count]
    held, length, rate = state.host("cell_quanta"), state.host("length"), state.host("rate")
    squeeze, gain, loss, crossed = (state.host(name) for name in ("squeeze", "gain", "loss", "crossed"))
    alive = kind_now != 0
    law = np.zeros(state.count, dtype=np.float32)
    for sort in (ENERGY_VOLUME, JOINT):
        mine = kind_now == sort
        law[mine] = (np.float64(state.size_energy[sort]) / held[mine].astype(np.float64)).astype(np.float32)
    kind, was = kind_now[:old], before["kind"]
    joints = (kind == JOINT) & (was == JOINT)
    volumes = (kind == ENERGY_VOLUME) & (was == ENERGY_VOLUME)
    change = held[:old] - before["held"]
    events = state.version - before["version"]
    lived = (kind != 0) & (was != 0)
    tick = state.host("clock")[:old] - before["clock"]
    all_volumes = kind_now == ENERGY_VOLUME
    most = np.flatnonzero(all_volumes)[np.argmax(held[all_volumes])]
    found.update(
        step=state.step, size_wrong=int((length[alive] != law[alive]).sum()),
        kind_wrong=int((state.host("kind") != kind_now).sum()), events=events,
        relinked=int(_links(state) != before["links"]), revived=int(((was == 0) & (kind != 0)).sum()),
        joint_fell=int((change[joints] < 0).sum()), joint_rose=int((change[joints] > 0).sum()),
        joint_rose_loose=int(((change > 0) & joints & ~(squeeze[:old] > 1.0)).sum()),
        joint_gain_wrong=int((change[joints] != gain[:old][joints]).sum()) if events == 0 else 0,
        joint_loss=int(loss[kind_now == JOINT].sum()),
        volume_wrong=int((change[volumes] != (gain[:old] - loss[:old])[volumes]).sum()) if events == 0 else 0,
        gained=int(gain[all_volumes].sum()), lost=int(loss[all_volumes].sum()), crossed_step=int(crossed.sum()),
        crossed_at_volumes=int(crossed[all_volumes].sum()),
        clock_wrong=int((np.abs(tick[lived] - settings.step_time * before["rate"][lived]) > 1e-4).sum())
        if events == 0 else 0,
        still=sum(int(np.array_equal(state.block(cell)[0], block)) for cell, block in zip(samples, before["blocks"])),
        size=float(length[alive].astype(np.float64).sum()), volumes=int(all_volumes.sum()),
        joints=int((kind_now == JOINT).sum()), collapses=state.collapses, merges=state.merges, tears=state.tears,
        quanta_max_is_shortest=int(length[most] == length[all_volumes].min()))
    return found
