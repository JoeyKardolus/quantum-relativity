"""The engine: a grid of cells built on the owner's seven axioms, one directory per axiom.

  axiom_1_energy/     one always-moving quantum fluid in whole signed quanta, exact books
  axiom_2_swirls/     every cell's fluid goes round its own axis; plus pulls, minus pushes
  axiom_3_structure/  the cells are the grid: relations and a size set by energy alone
  axiom_4_crossing/   the fluid's own step taken across a joint, only through joints;
                      the joint's friction takes direction, and that is heat
  axiom_5_packing/    more packed is smaller, slower and pulls harder; the pull is a wave
  axiom_6_limits/     squeezed joints make energy and collapse; overstretched volumes tear
  axiom_7_quantum/    phase everywhere; outflow that meets a fork splits, goes on and settles whole
Each holds `definitions/` (what things are), `logic/` (what happens in a step) and `tests/`.

This file is the only one that calls the logic of several axioms. It holds:
  ARRAYS, HOST   every axiom's state, merged (a name in two axioms raises)
  Settings       the six settings records in one frozen record; every number an axiom
                 leaves open is a field here, chosen before any run
  new_universe   the start: energies from the seed, sizes by the size law, nothing placed
  lay_start      the start at rest in balance: the pull at rest and every joint holding
                 the energy at which its push equals the pull across it
  step           one step: the axioms' logic called in a fixed order, nothing else
and the public names in `__all__`. Everything else is imported from its own file.
"""
from __future__ import annotations

from dataclasses import dataclass

from physics.engine.axiom_1_energy.definitions import energy_arrays
from physics.engine.axiom_1_energy.definitions.book_law import book_law
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.fill_joints import fill_joints
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_1_energy.logic.read_books import read_books
from physics.engine.axiom_1_energy.logic.start_energy import start_energy
from physics.engine.axiom_1_energy.logic.uneven_field import uneven_field
from physics.engine.axiom_2_swirls.definitions import swirl_arrays
from physics.engine.axiom_2_swirls.logic.start_swirls import start_swirls
from physics.engine.axiom_3_structure.definitions import structure_arrays
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, ENERGY_VOLUME, JOINT, NO_CELL
from physics.engine.axiom_3_structure.definitions.size_law import size_law_host
from physics.engine.axiom_3_structure.definitions.structure_settings import StructureSettings
from physics.engine.axiom_3_structure.logic.check_pattern import check_pattern
from physics.engine.axiom_3_structure.logic.measure_grid import measure_grid
from physics.engine.axiom_3_structure.logic.read_cells import read_cells
from physics.engine.axiom_3_structure.logic.read_ways import read_ways
from physics.engine.axiom_3_structure.logic.relink import relink
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_3_structure.logic.start_grid import start_grid
from physics.engine.axiom_4_crossing.definitions import crossing_arrays
from physics.engine.axiom_4_crossing.definitions.crossing_settings import CrossingSettings
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_5_packing.definitions import packing_arrays
from physics.engine.axiom_5_packing.definitions.packing_settings import PackingSettings
from physics.engine.axiom_5_packing.logic.balance_level import balance_level
from physics.engine.axiom_5_packing.logic.pack_cells import pack_cells
from physics.engine.axiom_5_packing.logic.press_joints import press_joints
from physics.engine.axiom_5_packing.logic.pull_fluid import pull_fluid
from physics.engine.axiom_5_packing.logic.rest_pull import rest_pull
from physics.engine.axiom_5_packing.logic.spread_pull import spread_pull
from physics.engine.axiom_5_packing.logic.start_pull import start_pull
from physics.engine.axiom_5_packing.logic.tick_clocks import tick_clocks
from physics.engine.axiom_6_limits.definitions import limits_arrays
from physics.engine.axiom_6_limits.definitions.limits_settings import LimitsSettings
from physics.engine.axiom_6_limits.logic.find_limits import find_limits
from physics.engine.axiom_6_limits.logic.make_minus import make_minus
from physics.engine.axiom_6_limits.logic.run_events import run_events
from physics.engine.axiom_6_limits.logic.start_fresh import start_fresh
from physics.engine.axiom_7_quantum.definitions import phase_arrays
from physics.engine.axiom_7_quantum.definitions.quantum_settings import QuantumSettings
from physics.engine.axiom_7_quantum.logic.follow import follow
from physics.engine.axiom_7_quantum.logic.meet_fork import meet_fork
from physics.engine.axiom_7_quantum.logic.react import react
from physics.engine.axiom_7_quantum.logic.start_store import start_store

__all__ = ["ARRAYS", "HOST", "Settings", "Grid", "new_universe", "lay_start", "step", "relink", "read_books", "book_law",
           "check_pattern", "measure_grid", "read_cells", "read_ways", "uneven_field", "react",
           "size_law_host", "ENERGY_VOLUME", "JOINT", "NO_CELL", "BLOCK", "BOXES"]

_STATE_FILES = (energy_arrays, swirl_arrays, structure_arrays, crossing_arrays, packing_arrays, limits_arrays,
                phase_arrays)


def _merged(table: str) -> dict:
    """Every axiom's `ARRAYS` or `HOST` in one dict; a name two axioms both claim raises."""
    merged: dict = {}
    for one in _STATE_FILES:
        for name, value in getattr(one, table).items():
            if name in merged:
                raise ValueError(f"{table} name {name!r} is claimed by two axioms")
            merged[name] = value
    return merged


_BALANCE_ROUNDS = 24
_RAISE_ROUNDS = 6
_BALANCED = 1e-4

ARRAYS = _merged("ARRAYS")
HOST = _merged("HOST")


@dataclass(frozen=True)
class Settings(EnergySettings, StructureSettings, CrossingSettings, PackingSettings, LimitsSettings,
               QuantumSettings):
    """Every number the axioms leave open, in one record: the six axioms' settings together."""

    def __post_init__(self) -> None:
        self.check_energy()
        self.check_structure()
        self.check_crossing()
        self.check_packing()
        self.check_limits()
        self.check_quantum()


def _resize(state, settings) -> None:
    """After quanta changed: every cell's count, then its size, then its packing and clock rate."""
    count_cells(state)
    size_cells(state, settings)
    pack_cells(state, settings)


def _balance(state, settings) -> None:
    """The start at rest in balance: every joint gets the energy at which its push equals the pull across it.

    A joint's energy sets its length, and the lengths set the pull, so the level is asked
    again (`balance_level`) until no joint moves by more than `_BALANCED` of what it holds, at most `_BALANCE_ROUNDS` times.
    Then, while any joint is still squeezed (at most `_RAISE_ROUNDS` times), every joint is raised by twice `_BALANCED`
    (a squeezed one from its level): the start is not squeezed anywhere and slack by a
    few times `_BALANCED` at most.
    `state.balance_off` keeps the last round's largest move.
    """
    for _ in range(_BALANCE_ROUNDS):
        press_joints(state, settings)
        held = state.host("cell_quanta").astype("float64")
        level = balance_level(state, settings)
        state.balance_off = fill_joints(state, settings, level)
        size_cells(state, settings)
        pack_cells(state, settings)
        rest_pull(state, settings)
        if state.balance_off <= _BALANCED:
            break
    for _ in range(_RAISE_ROUNDS):
        press_joints(state, settings)
        held = state.host("cell_quanta").astype("float64")
        level = balance_level(state, settings)
        if not (level > held).any():
            break
        fill_joints(state, settings, (level > 0.0) * (level * (level > held) + held * (level <= held)) * (1.0 + 2.0 * _BALANCED))
        size_cells(state, settings)
        pack_cells(state, settings)
        rest_pull(state, settings)


def lay_start(state: Grid, settings: Settings) -> None:
    """Lay a start at rest in balance for the energy as it lies: sizes, the pull at rest, balanced joints, the books.

    `new_universe` ends with it, and a scene that placed energy by hand in a fresh universe
    calls it again (the public door for a placed start). Raises ValueError after the first step.
    """
    if state.step != 0:
        raise ValueError(f"lay_start lays a start; this universe is at step {state.step}")
    _resize(state, settings)
    start_pull(state, settings)
    _balance(state, settings)
    start_fresh(state)
    start_store(state)
    state.start_books = read_books(state)


def new_universe(settings: Settings | None = None, cells_per_side: int = 32, seed: int = 1, spare: float = 0.25,
                 device: str = "cuda") -> Grid:
    """A fresh universe: the lattice, energies and swirls from the seed, laid at rest in balance."""
    settings = Settings() if settings is None else settings
    state = start_grid(cells_per_side, seed, spare, ARRAYS, HOST, device)
    start_energy(state, settings)
    start_swirls(state)
    lay_start(state, settings)
    return state


def step(state: Grid, settings: Settings, quantum: bool = True) -> None:
    """One step of the whole engine: each axiom's logic, called in the one fixed order.

    `quantum=False` is phase 1 of the fit (the bible, "The fit goes in phases"): the layers
    below the quantum one. Axiom 7's splits are then not opened, carried or settled; every
    other rule runs as it does with them. It is an argument of a run, never a setting.
    """
    clear_ledgers(state)
    tick_clocks(state, settings)
    spread_pull(state, settings)
    pull_fluid(state, settings)
    for axis in range(3):
        for parity in range(2):
            flow_inside(state, settings, axis, parity)
    cross(state, settings)
    count_cells(state)
    if quantum:
        meet_fork(state, settings)
    _resize(state, settings)
    press_joints(state, settings)
    make_minus(state, settings)
    _resize(state, settings)
    find_limits(state, settings)
    if state.found.numpy().any():
        relink(state, run_events(state, settings))
    count_cells(state)
    if quantum:
        follow(state)
    _resize(state, settings)
    state.step += 1
