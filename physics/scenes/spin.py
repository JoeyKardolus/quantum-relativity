"""Spin scene: one whole-step whirl in an energy volume, carried across a joint; how exact does it stay?

Contract: `spin(cells_per_side=8, boxes_per_cell_side=6, turns=1, drift_speed=0.15,
loop_half_width=2, track_reach=3.0, steps=60, settings=None, device="cuda")` starts
from new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0) (every
box holds 1) and imprints a phase (Universe.from_host's `phase`; the density stays
as it was, as a whirl is made in a lab):
  - the energy volume whose box centre lies nearest the grid's middle hosts the whirl;
    its joint with the most shared box faces sets the drift direction (the grid axis
    closest to the line from the volume's centre to the joint's), and the whirl's axis
    is the next grid axis, (drift + 1) % 3;
  - a whirl of `turns` whole turns round a straight line along that axis through the
    volume's centre, moved half a box off the boxes so no box sits on it; the line
    cannot end inside the fluid, so it runs through the whole column of cells along its
    axis, out to the open edge;
  - plus a phase rising drift_speed per box along the drift direction: the whole foam
    flows that way (the flow is the phase's gradient), carrying the whirl into the joint.
It runs `steps` steps (default settings Settings.all_on()), measuring at the start and
after every step, all from the complex field (wave_field.py) with physics.measure.
Cells meet only at their seams, so the phase may jump across a cell border and a square
of boxes straddling one says nothing about a whirl: only squares whose four boxes share
one cell count (phase_winding, the rest set to 0). From those: the whirl's local stretch
(squares within track_reach of the last centre, whirl_line; its along-axis place is held
at the start's); the winding inside the square loop of half width loop_half_width round
it (the windings of the counted squares it encloses, normal the whirl's axis, summed:
the loop's own winding wherever it lies in one cell); the loop's flow circulation
(loop_circulation); its strength (whirl_strength, a cylinder of radius and half height
loop_half_width, over the boxes of the cell holding its centre) and that cell.
Returns a dict (no verdict):
  cells_per_side, boxes_per_cell_side, steps, settings (dict), turns, drift_speed,
  whirl_axis, drift_axis (grid axes), drift_sign, start_cell, toward_joint (cell ids)
  time, found (1 while squares near the centre are pierced), winding (whole turns
  inside the loop), flow_circulation (the loop's flow circulation in whole steps),
  loop_in_one_cell (1 when every loop box lies in one cell, so the flow circulation is
  the whirl's alone), strength (angular momentum per unit
  density, whole steps), tilt (degrees, axis from the start's), azimuth (radians,
  unwrapped, the axis's lean round the start's axis: precession shows as a steady
  turn), drift (boxes moved along the drift direction), core_density (|psi|**2 at the
  box nearest the centre over the loop's mean), host_cell, host_kind (0 energy volume,
  1 joint, -1 empty) per sample
  winding_exact       1 when every sample's winding equals `turns`, else 0
  flow_offset_max     the largest |flow_circulation - turns| over the samples whose loop
                      lies in one cell (over all samples when none does)
  strength_min, strength_max, tilt_max, wobble_amplitude (standard deviation of tilt),
  wobble_period (the strongest period in the tilt, time units, by FFT; 0 when the tilt
  never changes), azimuth_turns (azimuth swept / 2 pi)
  crossed_joint       1 when a joint hosted the centre and an energy volume later did
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.measure import loop_circulation, phase_winding, whirl_line, whirl_strength
from physics.potts import ENERGY_VOLUME, JOINT, Settings, Universe, contact_pairs, new_universe
from physics.scenes.sampled_run import sampled_run
from physics.scenes.wave_field import wave_field


def spin(cells_per_side: int = 8, boxes_per_cell_side: int = 6, turns: int = 1, drift_speed: float = 0.15,
         loop_half_width: int = 2, track_reach: float = 3.0, steps: int = 60, settings: Settings | None = None,
         device: str = "cuda") -> dict:
    """Winding, flow circulation, strength and axis of one placed whirl, every step, as it drifts."""
    settings = settings or Settings.all_on()
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    owner, kind = even.owner.numpy(), even.kind.numpy()
    host, joint, centre = _host_and_joint(owner, kind)
    drift_axis = int(np.argmax(np.abs(centre[joint] - centre[host])))
    drift_sign = 1.0 if centre[joint][drift_axis] >= centre[host][drift_axis] else -1.0
    whirl_axis = (drift_axis + 1) % 3
    core = centre[host].copy()
    for axis in range(3):
        if axis != whirl_axis:
            core[axis] = math.floor(core[axis]) + 0.5
    phase = _whirl_phase(even.side, core, whirl_axis, turns) + drift_sign * drift_speed * np.arange(even.side,
        dtype=np.float64).reshape([-1 if axis == drift_axis else 1 for axis in range(3)])
    universe = Universe.from_host(owner, np.abs(wave_field(even)) ** 2, kind, boxes_per_cell_side, seed=1,
                                  device=device, phase=np.broadcast_to(phase, owner.shape))
    tracker = _Tracker(core, whirl_axis, drift_axis, drift_sign, loop_half_width, track_reach)
    samples = sampled_run(universe, settings, steps, 1, tracker.measure)
    result = {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
              "settings": asdict(settings), "turns": turns, "drift_speed": drift_speed, "whirl_axis": whirl_axis,
              "drift_axis": drift_axis, "drift_sign": drift_sign, "start_cell": int(host),
              "toward_joint": int(joint), "time": [time for time, _ in samples]}
    for name in samples[0][1]:
        result[name] = [measured[name] for _, measured in samples]
    result.update(_summary(result, turns))
    return result


def _host_and_joint(owner: np.ndarray, kind: np.ndarray) -> tuple[int, int, np.ndarray]:
    """The energy volume nearest the middle, its joint sharing the most faces, and every cell's centre."""
    occupied = owner >= 0
    places = np.argwhere(occupied).astype(np.float64)
    cells = owner[occupied]
    boxes = np.bincount(cells, minlength=kind.size).astype(np.float64)
    centre = np.stack([np.bincount(cells, weights=places[:, axis], minlength=kind.size) for axis in range(3)],
                      axis=1) / np.maximum(boxes, 1.0)[:, None]
    middle = (owner.shape[0] - 1) / 2.0
    volumes = np.flatnonzero((kind == ENERGY_VOLUME) & (boxes > 0))
    host = int(volumes[np.argmin(np.linalg.norm(centre[volumes] - middle, axis=1))])
    low, high, faces = contact_pairs(owner)
    touching = (low == host) | (high == host)
    others = np.where(low[touching] == host, high[touching], low[touching])
    shared = faces[touching]
    joints = kind[others] == JOINT
    return host, int(others[joints][np.argmax(shared[joints])]), centre


def _whirl_phase(side: int, core: np.ndarray, whirl_axis: int, turns: int) -> np.ndarray:
    """turns times the angle round the line through core along whirl_axis, right-handed."""
    across, up = (whirl_axis + 1) % 3, (whirl_axis + 2) % 3
    axis_values = np.arange(side, dtype=np.float64)
    first = (axis_values - core[across]).reshape([-1 if axis == across else 1 for axis in range(3)])
    second = (axis_values - core[up]).reshape([-1 if axis == up else 1 for axis in range(3)])
    return turns * np.arctan2(second, first)


class _Tracker:
    """Follows the whirl's centre from sample to sample; measure() reads every number off the field."""

    def __init__(self, core, whirl_axis, drift_axis, drift_sign, half_width, reach):
        self.start, self.centre = np.array(core), np.array(core)
        self.whirl_axis, self.drift_axis, self.drift_sign = whirl_axis, drift_axis, drift_sign
        self.half_width, self.reach = half_width, reach
        self.start_axis = np.eye(3)[whirl_axis]
        self.axis = self.start_axis.copy()

    def measure(self, universe: Universe) -> dict:
        field = wave_field(universe)
        owner = universe.owner.numpy()
        windings = _same_cell_windings(field, owner)
        line = whirl_line(windings, near=self.centre, reach=self.reach)
        if line["found"]:
            self.centre = np.array(line["centre"])
            self.centre[self.whirl_axis] = self.start[self.whirl_axis]
            if np.all(np.isfinite(line["axis"])):
                self.axis = np.array(line["axis"])
        loop = loop_circulation(field, self.centre, self.whirl_axis, self.half_width)
        nearest = tuple(np.rint(self.centre).astype(int))
        host = int(owner[nearest])
        loop_owners = _loop_square(owner, nearest, self.whirl_axis, self.half_width)
        ring = np.ones(loop_owners.shape, dtype=bool)
        ring[1:-1, 1:-1] = False
        lean = self.axis - (self.axis @ self.start_axis) * self.start_axis
        across, up = (self.whirl_axis + 1) % 3, (self.whirl_axis + 2) % 3
        density = np.abs(field) ** 2
        loop_density = _loop_mean_density(density, nearest, self.whirl_axis, self.half_width)
        return {"found": int(line["found"]),
                "winding": _enclosed_winding(windings, nearest, self.whirl_axis, self.half_width),
                "flow_circulation": loop["flow_circulation"],
                "loop_in_one_cell": int(np.all(loop_owners[ring] == loop_owners[ring][0])),
                "strength": whirl_strength(field, self.centre, self.axis, float(self.half_width),
                                           float(self.half_width), valid=(owner == host) & (host >= 0)),
                "tilt": float(np.degrees(np.arccos(np.clip(abs(self.axis @ self.start_axis), -1.0, 1.0)))),
                "azimuth": float(math.atan2(lean[up], lean[across])),
                "drift": float(self.drift_sign * (self.centre[self.drift_axis] - self.start[self.drift_axis])),
                "core_density": float(density[nearest] / loop_density) if loop_density > 0 else 0.0,
                "host_cell": host, "host_kind": int(universe.kind.numpy()[host]) if host >= 0 else -1}


def _same_cell_windings(field: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """phase_winding's windings, 0 on every square whose four boxes are not all one cell's."""
    windings = phase_winding(field, valid=owner >= 0)
    for axis in range(3):
        across, up = (axis + 1) % 3, (axis + 2) % 3
        corners = []
        for step_across, step_up in ((0, 0), (1, 0), (1, 1), (0, 1)):
            pick = [slice(None)] * 3
            pick[across] = slice(step_across, owner.shape[across] - 1 + step_across)
            pick[up] = slice(step_up, owner.shape[up] - 1 + step_up)
            corners.append(owner[tuple(pick)])
        one_cell = (corners[0] == corners[1]) & (corners[0] == corners[2]) & (corners[0] == corners[3])
        target = [slice(None)] * 3
        target[across] = slice(0, owner.shape[across] - 1)
        target[up] = slice(0, owner.shape[up] - 1)
        windings[axis][tuple(target)][~one_cell] = 0
    return windings


def _enclosed_winding(windings: np.ndarray, nearest: tuple, whirl_axis: int, half_width: int) -> int:
    """The counted squares' windings summed over the loop's inside, normal the whirl's axis (Stokes)."""
    across, up = (whirl_axis + 1) % 3, (whirl_axis + 2) % 3
    pick = [slice(None)] * 3
    pick[whirl_axis] = nearest[whirl_axis]
    pick[across] = slice(max(nearest[across] - half_width, 0), nearest[across] + half_width)
    pick[up] = slice(max(nearest[up] - half_width, 0), nearest[up] + half_width)
    return int(windings[whirl_axis][tuple(pick)].sum())


def _loop_square(values: np.ndarray, nearest: tuple, whirl_axis: int, half_width: int) -> np.ndarray:
    """The square of boxes half_width round the nearest box, in its layer (across, up)."""
    across, up = (whirl_axis + 1) % 3, (whirl_axis + 2) % 3
    pick = [slice(None)] * 3
    pick[whirl_axis] = nearest[whirl_axis]
    pick[across] = slice(nearest[across] - half_width, nearest[across] + half_width + 1)
    pick[up] = slice(nearest[up] - half_width, nearest[up] + half_width + 1)
    square = values[tuple(pick)]
    return square


def _loop_mean_density(density: np.ndarray, nearest: tuple, whirl_axis: int, half_width: int) -> float:
    """Mean density on the square ring of boxes half_width from the nearest box, in its layer."""
    square = _loop_square(density, nearest, whirl_axis, half_width)
    ring = np.ones(square.shape, dtype=bool)
    ring[1:-1, 1:-1] = False
    return float(square[ring].mean())


def _summary(result: dict, turns: int) -> dict:
    """Exactness, spread and wobble of the per-sample numbers."""
    tilt = np.array(result["tilt"])
    times = np.array(result["time"])
    kinds = result["host_kind"]
    joint_at = [index for index, kind in enumerate(kinds) if kind == JOINT]
    return {"winding_exact": int(all(winding == turns for winding in result["winding"])),
            "flow_offset_max": _flow_offset_max(result, turns),
            "strength_min": float(np.min(result["strength"])), "strength_max": float(np.max(result["strength"])),
            "tilt_max": float(tilt.max()), "wobble_amplitude": float(tilt.std()),
            "wobble_period": _strongest_period(times, tilt),
            "azimuth_turns": float((np.unwrap(result["azimuth"])[-1] - result["azimuth"][0]) / (2.0 * math.pi)),
            "crossed_joint": int(bool(joint_at) and ENERGY_VOLUME in kinds[joint_at[0]:])}


def _flow_offset_max(result: dict, turns: int) -> float:
    """The largest |flow circulation - turns| over the samples whose loop lies in one cell, else over all."""
    offsets = np.abs(np.array(result["flow_circulation"]) - turns)
    inside = np.array(result["loop_in_one_cell"]) == 1
    return float(offsets[inside].max() if inside.any() else offsets.max())


def _strongest_period(times: np.ndarray, values: np.ndarray) -> float:
    """The period of the largest non-constant Fourier term of evenly sampled values; 0 if they never change."""
    if values.size < 3 or np.ptp(values) == 0:
        return 0.0
    power = np.abs(np.fft.rfft(values - values.mean()))[1:]
    frequencies = np.fft.rfftfreq(values.size, d=float(times[1] - times[0]))[1:]
    return float(1.0 / frequencies[np.argmax(power)])
