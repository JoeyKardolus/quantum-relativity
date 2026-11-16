"""Belt trick scene: turn a whirl tied into the joint web once and twice; when does the web's twist return?

Contract: `belt_trick(cells_per_side=7, boxes_per_cell_side=10, margin=13, seed=1,
turns=1, drive_turns=2, steps_per_turn=60, loop_half_width=2, web_reach=8,
settings=None, device="cuda")` (bible `physics/quantum-relativity`, milestone 7's spin
lead: Dirac's belt trick). Its own settings, not the spin scene's: cells of 10 boxes and
an odd cells_per_side, so the host energy volume sits on the pull's centre (the whirl
survives both there, round-3 decisions 3 and 4); margin 13 makes the grid 96 boxes a side.
Start: physics.engine new_universe(cells_per_side, boxes_per_cell_side, seed,
unevenness=0, margin) (every box holds 1000 quanta); the energy volume (physics.engine
cells) whose box centre lies nearest the grid's middle is the host, H its start boxes. A
whirl of `turns` whole turns round a line along grid axis 2 through H's centre (half a
box off the boxes) is imprinted as phase on the whole grid (State.from_host's `phase`),
so the line runs through the column out to the open edge and the web round the host
carries the same winding: the whirl is tied in.
Two webs, each run twice (driven and still, so the web's own turning is told apart):
  tied     that start as it is: the host wrapped in joints, the joints one web
  untied   the control: every filled box one energy volume (no web): the same quanta,
           all made plus
Drive: turning the whirl's pattern by an angle a about its own line is turning its
phase by -turns a, so after each of the first drive_turns * steps_per_turn steps every
driven box's psi is multiplied by exp(-i turns 2 pi / steps_per_turn) (exact, norm
kept: the box's phase turns, its quanta stay): tied, the boxes of the energy volume
sharing the most boxes with H then; untied, H. The still runs skip it. The
host is turned about the whirl's own line only, never across it.
Measured at the start and after every step, from the complex field (wave_field.py):
  whirl_winding   whole turns inside the square loop of half width loop_half_width
                  round the line in H's middle layer (spin.py's counted squares:
                  squares whose four boxes share one cell)
  whirl_strength  whirl_strength over a cylinder of radius and half height
                  loop_half_width, driven boxes only
  web_winding     loop_circulation's winding round the line at half width
                  boxes_per_cell_side // 2 + 2 (a loop of joint boxes in the tied web)
  ring turn       for H (d = 0) and each closed surface of boxes at Chebyshev
                  distance d = 1..web_reach from H's bounding cube (joint boxes in the
                  tied web), arg sum psi(t) conj psi(0) over its occupied boxes: how
                  far that surface's fluid has turned since the start
  web_twist       the twist stored in the web, a configuration: the ring turns'
                  differences d to d + 1, each wrapped into (-1/2, 1/2] turn, summed
                  from H out to d = web_reach, in turns
  near_twist      the host's turn against the first surface's, wrapped, in turns
  tracked_twist   the host's turn against the outermost surface's, unwrapped from
                  sample to sample (history: it counts every turn the web did not follow)
  far_coherence   |sum psi(t) conj psi(0)| / sum |psi(t)| |psi(0)| on the outermost
                  surface: 1 when it is its start up to one common turn
Returns a dict (no verdict): cells_per_side, boxes_per_cell_side, margin, seed, turns,
drive_turns, steps_per_turn, loop_half_width, web_reach, settings (dict), host (cell id)
and runs: {"tied", "untied", "tied_still", "untied_still"}, each with time and every
measure per sample, plus at_turns: each measure at 0, 1, ... drive_turns turns.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.measure import loop_circulation, whirl_strength
from physics.engine import ENERGY_VOLUME, Settings, State, cells, host_boxes, lattice_of, new_universe, step
from physics.scenes.spin import _enclosed_winding, _same_cell_windings, _whirl_phase
from physics.scenes.wave_field import wave_field

WHIRL_AXIS = 2
MEASURES = ("whirl_winding", "whirl_strength", "web_winding", "web_twist", "near_twist", "tracked_twist",
            "far_coherence")


def belt_trick(cells_per_side: int = 7, boxes_per_cell_side: int = 10, margin: int = 13, seed: int = 1,
               turns: int = 1, drive_turns: int = 2, steps_per_turn: int = 60, loop_half_width: int = 2,
               web_reach: int = 8, settings: Settings | None = None, device: str = "cuda") -> dict:
    """The whirl's winding and strength and the web's twist, every step, while the host turns twice."""
    settings = settings or Settings()
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=seed, unevenness=0.0, margin=margin,
                        device=device)
    quanta = host_boxes(even)["quanta"]
    found = cells(host_boxes(even))
    host = _host(found)
    held = found["owner"] == host
    core = _box_centre(held)
    core[:WHIRL_AXIS] = np.floor(core[:WHIRL_AXIS]) + 0.5
    phase = np.broadcast_to(_whirl_phase(even.side, core, WHIRL_AXIS, turns), quanta.shape)
    rings = _rings(held, web_reach)
    starts = {"tied": lambda: State.from_host(quanta, phase, seed=seed, device=device, lattice=lattice_of(even)),
              "untied": lambda: State.from_host(np.abs(quanta), phase, seed=seed, device=device)}
    runs = {}
    for web, start in starts.items():
        for name, driven in ((web, drive_turns), (f"{web}_still", 0)):
            tracker = _Tracker(core, rings, web == "tied", held, loop_half_width, boxes_per_cell_side // 2 + 2)
            runs[name] = _run(start(), settings, tracker, turns, driven, steps_per_turn, drive_turns)
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "margin": margin,
            "seed": seed, "turns": turns, "drive_turns": drive_turns, "steps_per_turn": steps_per_turn,
            "loop_half_width": loop_half_width, "web_reach": web_reach, "settings": asdict(settings),
            "host": int(host), "runs": runs}


def _host(found: dict) -> int:
    """The energy volume whose box centre lies nearest the grid's middle."""
    volumes = np.flatnonzero(found["kind"] == ENERGY_VOLUME)
    middle = (found["owner"].shape[0] - 1) / 2.0
    return int(volumes[np.argmin(np.linalg.norm(found["centre"][volumes] - middle, axis=1))])


def _box_centre(mask: np.ndarray) -> np.ndarray:
    """The mean place of the masked boxes."""
    return np.argwhere(mask).astype(np.float64).mean(axis=0)


def _rings(held: np.ndarray, reach: int) -> list[np.ndarray]:
    """H, then the closed surfaces of boxes at Chebyshev distance 1..reach from H's bounding cube."""
    places = np.argwhere(held)
    low, high = places.min(axis=0), places.max(axis=0)
    axes = [np.arange(size) for size in held.shape]
    gaps = [np.maximum(np.maximum(low[axis] - axes[axis], axes[axis] - high[axis]), 0) for axis in range(3)]
    distance = np.maximum(np.maximum(gaps[0][:, None, None], gaps[1][None, :, None]), gaps[2][None, None, :])
    inside = np.zeros(held.shape, dtype=bool)
    inside[tuple(slice(low[axis], high[axis] + 1) for axis in range(3))] = True
    if reach < 1 or np.any(low - reach < 0) or np.any(high + reach >= np.array(held.shape)):
        raise ValueError(f"web_reach must be >= 1 and keep every surface inside the grid, got {reach}")
    return [held] + [(distance == ring) & ~inside for ring in range(1, reach + 1)]


def _run(universe: State, settings: Settings, tracker: "_Tracker", turns: int, driven_turns: int,
         steps_per_turn: int, drive_turns: int) -> dict:
    """Step drive_turns * steps_per_turn times, turning the host for driven_turns turns, measuring every step."""
    steps = drive_turns * steps_per_turn
    turn_per_step = turns * 2.0 * math.pi / steps_per_turn
    samples = [(0.0, tracker.measure(universe))]
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken <= driven_turns * steps_per_turn:
            _turn_phase(universe, tracker.driven(universe), -turn_per_step)
        samples.append((taken * settings.step_time, tracker.measure(universe)))
    run = {"time": [time for time, _ in samples]}
    for name in MEASURES:
        run[name] = [measured[name] for _, measured in samples]
    run["at_turns"] = {name: [run[name][count * steps_per_turn] for count in range(drive_turns + 1)]
                       for name in MEASURES}
    return run


def _turn_phase(universe: State, boxes: np.ndarray, angle: float) -> None:
    """Multiply psi by exp(i angle) on the given boxes: their phase turns, their whole quanta stay."""
    phase = universe.phase.numpy().astype(np.float64)
    phase[boxes] = np.angle(np.exp(1j * (phase[boxes] + angle)))
    universe.phase.assign(phase.astype(np.float32))


class _Tracker:
    """Holds the start's field on the surfaces; measure() reads every number off the field now."""

    def __init__(self, core, rings, tied, held, loop_half_width, web_half_width):
        self.core, self.rings, self.tied, self.held = np.array(core), rings, tied, held
        self.loop_half_width, self.web_half_width = loop_half_width, web_half_width
        self.start = None
        self.tracked, self.last_raw = 0.0, None

    def driven(self, universe: State, found: dict | None = None) -> np.ndarray:
        """The boxes turned by the drive: the energy volume sharing most boxes with H now (tied), else H."""
        if not self.tied:
            return self.held
        found = found or cells(host_boxes(universe))
        owner = found["owner"]
        shared = owner[self.held & (owner >= 0) & (owner < found["volumes"])]
        if shared.size == 0:
            return np.zeros(self.held.shape, dtype=bool)
        return owner == np.bincount(shared).argmax()

    def measure(self, universe: State) -> dict:
        field = wave_field(universe)
        if self.start is None:
            self.start = field.copy()
        found = cells(host_boxes(universe))
        owner = found["owner"]
        nearest = tuple(np.rint(self.core).astype(int))
        driven = self.driven(universe, found) & (owner >= 0)
        windings = _same_cell_windings(field, owner)
        strength = (whirl_strength(field, self.core, np.eye(3)[WHIRL_AXIS], float(self.loop_half_width),
                                   float(self.loop_half_width), valid=driven) if driven.any() else 0.0)
        ring_turns = [self._ring_turn(field, ring & (owner >= 0)) for ring in self.rings]
        steps = [_wrap(ring_turns[index] - ring_turns[index + 1]) for index in range(len(ring_turns) - 1)]
        raw = ring_turns[0] - ring_turns[-1]
        if self.last_raw is not None:
            self.tracked += _wrap(raw - self.last_raw)
        self.last_raw = raw
        outer = self.rings[-1] & (owner >= 0)
        return {"whirl_winding": _enclosed_winding(windings, nearest, WHIRL_AXIS, self.loop_half_width),
                "whirl_strength": float(strength),
                "web_winding": _web_winding(field, self.core, self.web_half_width),
                "web_twist": float(sum(steps)), "near_twist": float(steps[0]),
                "tracked_twist": float(self.tracked),
                "far_coherence": _coherence(field[outer], self.start[outer])}

    def _ring_turn(self, field: np.ndarray, ring: np.ndarray) -> float:
        """How far a surface's fluid has turned since the start, in turns, wrapped into (-1/2, 1/2]."""
        overlap = np.sum(field[ring] * np.conj(self.start[ring]))
        return float(np.angle(overlap) / (2.0 * math.pi)) if abs(overlap) > 0 else 0.0


def _web_winding(field: np.ndarray, core: np.ndarray, half_width: int) -> int:
    """The phase winding round the loop of half width half_width about the line; 0 if a loop box is empty."""
    try:
        return int(loop_circulation(field, core, WHIRL_AXIS, half_width)["winding"])
    except ValueError:
        return 0


def _coherence(now: np.ndarray, start: np.ndarray) -> float:
    """|sum now conj(start)| / sum |now| |start|: 1 when now is start up to one common turn."""
    scale = float(np.sum(np.abs(now) * np.abs(start)))
    return float(abs(np.sum(now * np.conj(start))) / scale) if scale > 0 else 0.0


def _wrap(turn: float) -> float:
    """A turn wrapped into (-1/2, 1/2]."""
    return float(-((-turn + 0.5) % 1.0 - 0.5))
