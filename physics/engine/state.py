"""State: the engine's boxes on the GPU and its graph of cells, universes side by side.

Contract: `State.from_host(quanta, phase, seed=1, device=None, lattice=None, wrap=False)`
copies signed quanta and phases (radians) and the cell graph to the GPU and returns the
state step() advances. `quanta` is a cube (side, side, side) for one universe, or
(universes, side, side, side) for a batch stepped side by side; `phase` has the same
shape. Universe u holds first indices u * side .. u * side + side - 1 of every per-box
array; no rule reads from one universe into the next (reach.py). `wrap` True makes
every universe wrap round at its edges (opposite faces meet, no walls; an even side).
Per box:
  quanta     int32 whole quanta: plus = an energy volume's box, minus = a joint's box,
             0 = no space. The amount ignoring sign is conserved exactly (bible rule 4)
  phase      float32 in (-pi, pi], the quantum fluid's phase; the fluid's complex number
             is sqrt(|quanta|) exp(i phase), never stored
  pull       float32 the pull (energy per plus quantum), a wave at light speed
  pull_rate  float32 its time derivative (leapfrog, pull_wave.py)
  clock      float32 the proper time the box has lived, at its cell's clock rate
  owner      int32 the cell the box belongs to, -1 for empty space
  marks      uint8 per step which rule changed the box: 1 collapse, 2 owner change, 8 split
The cells are a graph (owner, 2026-11-16): each cell keeps its own neighbours, so one
volume can split or two merge without touching any other cell. Per universe and cell id
(capacity `capacity`, the same for every universe; ids of cells that do not exist have
kind 0), on the host (numpy, the cell rules run there, cell_laws.py):
  kind        int8 1 energy volume, -1 joint, 0 no cell
  between     int32 (.., 2) a joint's two energy volumes (lower id first; -1 for a side
              that faces empty space or a wall); -1 for a volume
  place       int32 (.., 3) the start lattice place of a cell new_universe built, -1
              for a cell made later (a split's) or a start without a lattice
  size        float64 the size in boxes the cell's rule gives it: an energy volume's is
              its wanted size (size_energy[0] / quanta) ** 3 (bible rule 4, length =
              constant / energy), a joint's swings round the size its energy wants
              (joint_swing.py); the boxes follow it (exchange.py)
  speed       float64 a joint's size change per unit of its local time
  wanted      float64 the size the cell's energy wants, (size_energy[k] / quanta) ** 3
  cell_push   float64 a joint's push per quantum, push * (wanted / size) ** steepness;
              0 for a volume (a volume only pulls)
  rate        float64 the cell's clock rate, 1 / (1 + packing / clock_packing)
  cell_clock  float64 the proper time the cell has lived
  cell_quanta int64 and cell_boxes int64: the cell's quanta (ignoring sign) and boxes,
              counted at the start of every step (cell_count.py)
  crossed     int64 per joint the quanta that crossed it this step from between[0] to
              between[1] (negative: the other way; cross.py)
size_energy float64 (universes, 2): the constant of length = constant / energy for energy
volumes and for joints, fixed once by the start's correction (new_universe) or, when a
start does not give it, so that the start's cells want on average the room they hold.
tallies int64 (universes, 4): quanta crossed joints (both ways), joint boxes collapsed,
owner changes, minus quanta collapse turned plus. splits, torn, collapses (numpy, per
universe): volume splits and the plus quanta their tears turned minus, joint collapses.
`lattice` is None (the cells are read off the boxes: every 26-connected group of plus
boxes one energy volume, every face-connected group of minus boxes one joint between the
volumes it touches) or a dict, or one dict per universe, with "owner" (the box array),
"kind", "between" and optionally "place", "size", "speed", "size_energy" (2,) and
"cell_clock" per cell id, as host_boxes / lattice_of give them. Every box's sign must
match its owner's kind (ValueError otherwise). `seed` is an int (universe u gets seed +
u) or one int per universe; `keys` hash them for the rounding (a universe steps the same
alone as side by side). `start` holds per universe the start's amounts and boxes.
`ladder` is None or the detail ladder's store (ladder_sync.py). Raises ValueError on a
non-cubic or mismatched start, a seed count that is not one per universe, a box of 2**30
quanta or more, or an odd side with wrap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp
from scipy import ndimage

from physics.engine.group_label import group_label

_HEADROOM = 64


@dataclass
class State:
    """The engine's per-box arrays on the GPU, its cell graph, its universes' seeds and its step counter."""

    side: int
    universes: int
    wrap: bool
    quanta: wp.array3d
    phase: wp.array3d
    pull: wp.array3d
    pull_rate: wp.array3d
    clock: wp.array3d
    owner: wp.array3d
    marks: wp.array3d
    tallies: wp.array
    seeds: tuple
    keys: wp.array
    kind: np.ndarray
    between: np.ndarray
    place: np.ndarray
    size_energy: np.ndarray
    size: np.ndarray
    speed: np.ndarray
    cell_clock: np.ndarray
    start: tuple = ()
    step: int = 0
    ladder: object = None
    substeps: object = None
    capped: object = None
    splits: object = None
    torn: object = None
    collapses: object = None
    wanted: object = None
    cell_push: object = None
    rate: object = None
    cell_quanta: object = None
    cell_boxes: object = None
    crossed: object = None
    device: object = None
    cells_d: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        """The per-box arrays' shape: (universes * side, side, side)."""
        return (self.universes * self.side, self.side, self.side)

    @property
    def capacity(self) -> int:
        """Cell ids per universe."""
        return int(self.kind.shape[1])

    def universe(self, index: int) -> slice:
        """The first-axis slice universe `index` holds."""
        if not 0 <= index < self.universes:
            raise ValueError(f"universe {index} is not one of {self.universes}")
        return slice(index * self.side, (index + 1) * self.side)

    def joints_of(self, universe: int, volume: int) -> np.ndarray:
        """The ids of the joints an energy volume has (its neighbours in the graph)."""
        sides = self.between[universe]
        return np.flatnonzero((self.kind[universe] == -1) & ((sides[:, 0] == volume) | (sides[:, 1] == volume)))

    def new_cell(self, universe: int, kind: int) -> int:
        """A fresh cell id of `kind` in one universe, growing the capacity when needed."""
        fresh = np.flatnonzero((self.kind[universe] == 0) & (self.place[universe, :, 0] < 0))
        if fresh.size == 0:
            self._grow(_HEADROOM)
            return self.new_cell(universe, kind)
        cell = int(fresh[0])
        self.kind[universe, cell] = kind
        self.between[universe, cell] = -1
        self.place[universe, cell] = -1
        for name in ("size", "speed", "cell_clock"):
            getattr(self, name)[universe, cell] = 0.0
        return cell

    def _grow(self, more: int) -> None:
        """Add `more` cell ids to every universe's table."""
        def widen(values, fill):
            extra = np.full((values.shape[0], more) + values.shape[2:], fill, dtype=values.dtype)
            return np.concatenate([values, extra], axis=1)
        self.kind = widen(self.kind, 0)
        self.between = widen(self.between, -1)
        self.place = widen(self.place, -1)
        for name in ("size", "speed", "cell_clock"):
            setattr(self, name, widen(getattr(self, name), 0.0))
        self.cells_d = {}

    @classmethod
    def from_host(cls, quanta: np.ndarray, phase: np.ndarray, seed=1, device: str | None = None,
                  lattice=None, wrap: bool = False) -> "State":
        """Copy signed quanta, phases and the cell graph to the GPU; pull, clocks and tallies start at 0."""
        quanta = np.asarray(quanta)
        phase = np.asarray(phase)
        if quanta.ndim == 3:
            quanta, phase = quanta[None], phase[None] if phase.ndim == 3 else phase
        if quanta.ndim != 4 or len(set(quanta.shape[1:])) != 1:
            raise ValueError(f"quanta must be a cube or a batch of cubes, got shape {quanta.shape}")
        if phase.shape != quanta.shape:
            raise ValueError(f"phase shape {phase.shape} differs from quanta shape {quanta.shape}")
        if np.abs(quanta.astype(np.int64)).max(initial=0) >= 2 ** 30:
            raise ValueError("a box may hold fewer than 2**30 quanta (int32 headroom for moves)")
        universes, side = int(quanta.shape[0]), int(quanta.shape[1])
        if wrap and side % 2:
            raise ValueError("a universe that wraps round needs an even side")
        seeds = _seeds(seed, universes)
        graphs = [_graph(quanta[index], _one(lattice, index), wrap) for index in range(universes)]
        capacity = max(graph["kind"].size for graph in graphs) + _HEADROOM
        stacked = (universes * side, side, side)
        keys = np.array([(value * 2654435761 + 12345) & 0x7FFFFFFF for value in seeds], dtype=np.int32)
        owner = np.concatenate([graph["owner"] for graph in graphs]).reshape(stacked)
        state = cls(side=side, universes=universes, wrap=bool(wrap),
                    quanta=wp.array(quanta.astype(np.int32).reshape(stacked), dtype=wp.int32, device=device),
                    phase=wp.array(_wrapped(phase).reshape(stacked), dtype=wp.float32, device=device),
                    pull=wp.zeros(stacked, dtype=wp.float32, device=device),
                    pull_rate=wp.zeros(stacked, dtype=wp.float32, device=device),
                    clock=wp.zeros(stacked, dtype=wp.float32, device=device),
                    owner=wp.array(owner.astype(np.int32), dtype=wp.int32, device=device),
                    marks=wp.zeros(stacked, dtype=wp.uint8, device=device),
                    tallies=wp.zeros((universes, 4), dtype=wp.int64, device=device),
                    seeds=seeds, keys=wp.array(keys, dtype=wp.int32, device=device),
                    kind=_table(graphs, "kind", capacity, 0, np.int8),
                    between=_table(graphs, "between", capacity, -1, np.int32),
                    place=_table(graphs, "place", capacity, -1, np.int32),
                    size_energy=np.stack([graph["size_energy"] for graph in graphs]),
                    size=_table(graphs, "size", capacity, 0.0, np.float64),
                    speed=_table(graphs, "speed", capacity, 0.0, np.float64),
                    cell_clock=_table(graphs, "cell_clock", capacity, 0.0, np.float64),
                    start=tuple(_start(cube) for cube in quanta),
                    substeps=wp.array(np.ones(universes, dtype=np.int32), dtype=wp.int32, device=device),
                    capped=np.zeros(universes, dtype=np.int64), splits=np.zeros(universes, dtype=np.int64),
                    torn=np.zeros(universes, dtype=np.int64), collapses=np.zeros(universes, dtype=np.int64),
                    device=device)
        for index, graph in enumerate(graphs):
            state.start[index]["cells"] = int((graph["kind"] != 0).sum())
        return state


def _one(lattice, index: int):
    """The universe's own graph dict, or None."""
    if lattice is None or isinstance(lattice, dict):
        return lattice
    return lattice[index]


def _table(graphs: list, name: str, capacity: int, fill, dtype) -> np.ndarray:
    """One per-cell table over the batch, padded to the capacity."""
    first = np.asarray(graphs[0][name])
    table = np.full((len(graphs), capacity) + first.shape[1:], fill, dtype=dtype)
    for index, graph in enumerate(graphs):
        values = np.asarray(graph[name])
        table[index, :values.shape[0]] = values
    return table


def _graph(cube: np.ndarray, lattice: dict | None, wrap: bool) -> dict:
    """One universe's checked cell graph: owner per box, kind, between, place, sizes, size constants."""
    graph = _read_off(cube, wrap) if lattice is None else _given(lattice, cube.shape)
    owner = graph["owner"]
    occupied = owner >= 0
    kinds = np.asarray(graph["kind"], dtype=np.int8)
    if owner.max(initial=-1) >= kinds.size:
        raise ValueError(f"an owner lies outside the {kinds.size} cells of the graph")
    if np.any(occupied != (cube != 0)) or not np.array_equal(kinds[owner[occupied]], np.sign(cube[occupied])):
        raise ValueError("every box's sign must match its owner's kind (plus volume, minus joint, 0 no cell)")
    amounts = np.bincount(owner[occupied], weights=np.abs(cube[occupied]).astype(np.float64), minlength=kinds.size)
    boxes = np.bincount(owner[occupied], minlength=kinds.size).astype(np.float64)
    if "size_energy" not in graph:
        graph["size_energy"] = np.array([_constant(amounts, boxes, kinds == sign) for sign in (1, -1)])
    if "size" not in graph:
        graph["size"] = boxes
    graph.setdefault("speed", np.zeros(kinds.size))
    graph.setdefault("cell_clock", np.zeros(kinds.size))
    graph.setdefault("place", np.full((kinds.size, 3), -1, dtype=np.int32))
    graph["kind"] = kinds
    graph["owner"] = owner.astype(np.int32)
    return graph


def _given(lattice: dict, shape: tuple) -> dict:
    """A graph handed in by the caller, copied."""
    graph = {name: np.array(value) for name, value in lattice.items() if name in (
        "owner", "kind", "between", "place", "size", "speed", "size_energy", "cell_clock")}
    graph["owner"] = np.broadcast_to(graph["owner"], shape).astype(np.int32)
    return graph


def _constant(amounts: np.ndarray, boxes: np.ndarray, members: np.ndarray) -> float:
    """The constant C of length = C / energy with which the cells of one kind want, summed, the boxes they hold."""
    held = members & (amounts > 0)
    if not held.any():
        return 1.0
    return float(np.cbrt(boxes[held].sum() / (amounts[held] ** -3.0).sum()))


def _read_off(cube: np.ndarray, wrap: bool) -> dict:
    """Cells read off the signs: each 26-connected plus group a volume, each face-connected minus group a joint."""
    plus, volumes = group_label(cube > 0, wrap, True)
    minus, joints = group_label(cube < 0, wrap)
    owner = np.full(cube.shape, -1, dtype=np.int32)
    owner[cube > 0] = plus[cube > 0] - 1
    owner[cube < 0] = minus[cube < 0] - 1 + volumes
    kind = np.concatenate([np.ones(volumes, dtype=np.int8), -np.ones(joints, dtype=np.int8)])
    between = np.full((volumes + joints, 2), -1, dtype=np.int32)
    for joint in range(joints):
        near = ndimage.binary_dilation(minus == joint + 1) & (cube > 0)
        touching = np.unique(plus[near]) - 1
        if touching.size:
            between[volumes + joint, :min(2, touching.size)] = touching[:2]
    return {"owner": owner, "kind": kind, "between": _ordered(between)}


def _ordered(between: np.ndarray) -> np.ndarray:
    """Each row's real ids first in rising order, -1 after."""
    rows = np.where(between < 0, np.iinfo(np.int32).max, between)
    rows = np.sort(rows, axis=1)
    return np.where(rows == np.iinfo(np.int32).max, -1, rows).astype(np.int32)


def _wrapped(phase: np.ndarray) -> np.ndarray:
    """Phases brought into [-pi, pi) as float32, a zero array copied without arithmetic (large starts)."""
    angles = np.asarray(phase, dtype=np.float32)
    if not angles.any():
        return angles.copy() if angles is phase else angles
    return (np.remainder(angles.astype(np.float64) + np.pi, 2.0 * np.pi) - np.pi).astype(np.float32)


def _seeds(seed, universes: int) -> tuple:
    """One int seed per universe: seed + u for an int, else the given sequence."""
    if np.ndim(seed) == 0:
        return tuple(int(seed) + index for index in range(universes))
    seeds = tuple(int(value) for value in seed)
    if len(seeds) != universes:
        raise ValueError(f"need one seed per universe ({universes}), got {len(seeds)}")
    return seeds


def _start(cube: np.ndarray) -> dict:
    """One universe's start reference: its amounts and occupied boxes."""
    plus = cube[cube > 0].astype(np.int64)
    minus = -cube[cube < 0].astype(np.int64)
    return {"amount": int(plus.sum() + minus.sum()), "plus": int(plus.sum()), "minus": int(minus.sum()),
            "boxes": int(plus.size + minus.size), "plus_boxes": int(plus.size), "cells": None}
