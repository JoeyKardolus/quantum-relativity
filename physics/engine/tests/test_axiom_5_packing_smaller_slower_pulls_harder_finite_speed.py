"""Axiom 5, Packing: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker with distance; squeezed minus pushes back steeply.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16), checked in 6 parts, each
kept whole from the file it came from (names that two parts shared carry a _P<n> suffix).

Part 1 (was axiom 06). More energy makes a cell smaller, across the cells at every check.

Bible rule 3 and rule 4 (length = constant / energy). Default universe (seed 1,
Settings(), axiom_run.py), every 10 steps from step 10 to 300 (at step 0 every cell holds
the same 1000 boxes by construction, so no rank order exists): the Spearman rank
correlation of a cell's quanta against its box count is strongly negative, at most
-0.7, among the energy volumes and among the joints, each kind on its own (their sizes
follow one law but the two kinds are not compared with each other). Threshold set
before the run.

Part 2 (was axiom 08). Pull spreads through all space at a finite speed, weaker with distance; a smaller packed volume pulls harder.

Bible rule 7 and rule 2 (owner, 2026-11-16). Measured:
  1. default universe (seed 1, Settings(), axiom_run.py), every 10 steps for 300 steps:
     no box holds a nonzero pull farther beyond the plus boxes' bounding box than the
     steps taken plus how far the plus boxes themselves spread (finite speed: at most
     one box per step on the wave's stencil), and from step 100 on (the wave has
     crossed the filled cube) at least 99% of the occupied boxes hold a nonzero pull
     (it reaches all of space)
  2. staged, pull wave only (pull_wave.py, Settings()): a 64-box cube of joint boxes
     (-1000, they carry the pull but add none) with the same 512,000 plus quanta at its
     centre either packed in 4**3 boxes or spread over 8**3 boxes. Over steps 120..240
     (after the front has reached the walls) the pull's time average along the x axis
     from the centre: after n <= 60 steps no box farther than n boxes from the source
     holds any pull; outside the source the averaged pull gets weaker with distance
     at every box from 8 to 28 boxes out; and at the centre the packed source's
     averaged pull is deeper than the spread one's

Part 3 (was axiom 13). Everything inside a cell (pull, resizing, flow, collapse) runs on that cell's local clock, slower where more packed.

Bible "Time is local" and audit answers (owner, 2026-11-16: everything inside a cell
runs on that cell's local clock). Measured:
  1. default universe (seed 1, Settings(), axiom_run.py) at step 300: the cells' local
     time lived (each cell's own clock) against their packing (quanta per box), each
     kind on its own: Spearman rank correlation at most -0.7 (more packed, less time)
  2. side by side, one step of new_universe(5, 10, seed=1, margin=8) (uneven) with
     Settings() and with local clocks off (clock_packing 0, a comparison run only): a
     rule that runs on local clocks gives a different result when the clocks are off.
     After the one step the flow (quanta and phases), the pull and the cells' sizes
     (resizing, state.size) each differ between the two runs. Collapse is a threshold acting at
     once, so a clock has nothing to slow in it in this engine; it is not measured

Part 4 (was test_the_pull_arrives_later_farther_away_at_light_speed). The pull spreads box to box at light speed: a source's pull arrives later the farther away, at the set speed.

Issue #3: pull spreads at a fixed finite speed instead of being solved across all space
at once. A grid of 96 boxes a side is all joint (-1000 per box: space that carries the
pull and adds none) with a ball of radius 2 boxes of plus quanta (+1000 per box) at its
middle, switched on at step 0. Only `pull_wave` runs (default settings, light speed 1
box per unit time, step_time 0.25, with local clocks off: the pull runs on each cell's
clock since 2026-11-16, and this measures the wave's own speed; axiom 5 measures the
clocks). At distances r = 10, 14, ..., 34 boxes along +x the
arrival is the first step at which the pull there reaches half its static value
-G M / r. The arrival times rise with r, and the slope of a straight-line fit of r
against arrival time is the light speed within 10%. The walls (48 boxes out) reflect,
but their echo reaches r = 34 only after its arrival. The same is measured along the
body diagonal and printed.

Part 5 (was test_the_pull_is_the_same_in_every_direction_within_five_percent). The pull is direction-free: behind its front it is -G M / r along an axis, a face diagonal and a body diagonal alike, within 5%.

Issue #3 acceptance (direction-free within 5%). A 96-box grid is all joint (-1000 per
box: space that carries the pull and adds none) with a ball of radius 2 boxes of plus
quanta (+1000 per box) at its middle, switched on at step 0. Only `pull_wave` runs
(default settings: light speed 1, step_time 0.25) for 160 steps, 40 time units. For a
source switched on at t = 0 the wave equation's pull inside the light cone is the static
-G M / r (Kirchhoff), so at r = 8..16 boxes, well behind the front at 40 and before the
walls' echo (back at r = 16 only at t = 64), the pull is compared with -G M / r. Each
direction's mean ratio over its lattice points, and every single point, lies within 5% of
1, and the three directions' means within 5% of each other.

Part 6 (2026-11-17). Squeezed minus pushes back as steeply as the setting says, in the swing too.

A joint's push is push * (wanted size / size) ** push_steepness (settings.py), and the
joints swing on that push without friction (joint_swing.py), so one swing must keep
the swing energy written with that same steepness (swing_energy.py). Before 2026-11-17
the swing had 6 written in whatever the setting said, so a search over push_steepness
moved nothing. new_universe(8, 10, seed 1), 20 steps of the whole engine with
push_steepness 3.5 and with 8.25, then one more swing with the pull as load: the energy
changes by less than 1e-9 of itself, and the same swing's energy written with steepness
6 changes by more than 1e-6 of itself (the two steepnesses are told apart). Thresholds
set before the run.
"""
import numpy as np

from scipy.stats import spearmanr
from physics.engine.tests._gpu import needs_gpu
from physics.engine import Settings, State, cell_count, cell_laws, joint_swing, new_universe, pull_wave, step
from physics.engine.tests.swing_energy import swing_energy


MOST_P1 = -0.7


@needs_gpu
def test_cells_with_more_quanta_hold_fewer_boxes_at_every_check():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    volume_rho, joint_rho = [], []
    for check in run["checks"][1:]:
        found = check["cells"]
        count = found["volumes"]
        volume_rho.append(spearmanr(found["quanta"][:count], found["boxes"][:count])[0])
        joint_rho.append(spearmanr(found["quanta"][count:], found["boxes"][count:])[0])
    print("volumes rho by check: " + " ".join(f"{value:+.2f}" for value in volume_rho))
    print("joints  rho by check: " + " ".join(f"{value:+.2f}" for value in joint_rho))
    print(f"volumes {min(volume_rho):+.2f} .. {max(volume_rho):+.2f}; joints {min(joint_rho):+.2f} .. "
          f"{max(joint_rho):+.2f} (need every one <= {MOST_P1})")
    assert max(volume_rho) <= MOST_P1 and max(joint_rho) <= MOST_P1


SIDE_P2 = 64


CENTRE_P2 = 32


def _source(across: int) -> np.ndarray:
    """Joint boxes everywhere inside a 1-box empty rim, 512,000 plus quanta in a cube `across` wide at the centre."""
    quanta = np.zeros((SIDE_P2,) * 3, dtype=np.int32)
    quanta[1:-1, 1:-1, 1:-1] = -1000
    low = CENTRE_P2 - across // 2
    quanta[low:low + across, low:low + across, low:low + across] = 512000 // across ** 3
    return quanta


def _staged(across: int) -> tuple:
    """(farthest pull reach beyond the source at steps 10..60 minus the steps, time-averaged pull along +x)."""
    quanta = _source(across)
    state = State.from_host(quanta, np.zeros(quanta.shape))
    settings = Settings()
    low = CENTRE_P2 - across // 2
    overshoot = -10 ** 9
    total = np.zeros(SIDE_P2 - CENTRE_P2)
    for count in range(1, 241):
        pull_wave(state, settings)
        if count <= 60 and count % 10 == 0:
            places = np.argwhere(state.pull.numpy() != 0)
            beyond = np.maximum(np.maximum(low - places, places - (low + across - 1)), 0).max()
            overshoot = max(overshoot, int(beyond) - count)
        if count >= 120:
            total += state.pull.numpy()[CENTRE_P2:, CENTRE_P2, CENTRE_P2]
    return overshoot, total / 121.0


@needs_gpu
def test_the_pull_has_a_finite_speed_reaches_all_space_fades_and_is_deeper_from_a_packed_source():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    overshoot = max(check["pull_reach"] - check["step"] - check["plus_spread"] for check in run["checks"][1:])
    cover = min(check["pull_cover"] for check in run["checks"] if check["step"] >= 100)
    packed_reach, packed = _staged(4)
    spread_reach, spread = _staged(8)
    outside = np.abs(packed[8:29])
    fades = bool(np.all(np.diff(outside) < 0))
    print(f"default run: pull reach beyond plus boxes minus (steps + plus spread), worst {overshoot}; least share of "
          f"occupied boxes with pull from step 100: {cover:.4f}")
    print(f"staged: reach past n boxes after n steps, worst {max(packed_reach, spread_reach)}; averaged pull at "
          f"centre packed {packed[0]:.4g}, spread {spread[0]:.4g}; |pull| from 8 to 28 boxes out: "
          + " ".join(f"{value:.3g}" for value in outside) + f"; strictly fading {fades}")
    assert overshoot <= 0 and max(packed_reach, spread_reach) <= 0
    assert cover >= 0.99
    assert fades
    assert packed[0] < spread[0] < 0


MOST_P3 = -0.7


@needs_gpu
def test_more_packed_cells_live_less_time():
    from physics.engine.tests.axiom_run import axiom_run
    found = axiom_run()["checks"][-1]["cells"]
    count = found["volumes"]
    packing = found["quanta"] / np.maximum(found["boxes"], 1)
    volume_rho = spearmanr(packing[:count], found["clock"][:count])[0]
    joint_rho = spearmanr(packing[count:], found["clock"][count:])[0]
    print(f"step 300: local time vs packing, volumes rho {volume_rho:+.3f}, joints rho {joint_rho:+.3f} (need <= {MOST_P3}); "
          f"local time {found['clock'].min():.3f} .. {found['clock'].max():.3f} of counter time 75")
    assert volume_rho <= MOST_P3 and joint_rho <= MOST_P3


@needs_gpu
def test_pull_resizing_and_flow_all_follow_the_local_clocks():
    results = []
    for packing in (Settings().clock_packing, 0.0):
        state = new_universe(5, 10, seed=1, margin=8)
        step(state, Settings(clock_packing=packing))
        results.append({name: np.asarray(getattr(state, name).numpy() if hasattr(getattr(state, name), "numpy")
                                         else getattr(state, name)) for name in ("quanta", "phase", "pull", "size")})
    gaps = {name: float(np.abs(results[0][name].astype(np.float64) - results[1][name].astype(np.float64)).max())
            for name in results[0]}
    print("clocks on vs off after one step, largest difference: "
          + ", ".join(f"{name} {value:.4g}" for name, value in gaps.items()))
    assert gaps["quanta"] > 0 or gaps["phase"] > 0
    assert gaps["pull"] > 0
    assert gaps["size"] > 0


SIDE_P4 = 96


CENTRE_P4 = 48


DISTANCES = np.arange(10, 35, 4)


@needs_gpu
def test_the_pull_arrives_at_r_over_c():
    settings = Settings(clock_packing=0.0)
    places = np.arange(SIDE_P4) - CENTRE_P4
    radius = np.sqrt(places[:, None, None] ** 2 + places[None, :, None] ** 2 + places[None, None, :] ** 2)
    quanta = np.where(radius <= 2.0, 1000, -1000).astype(np.int32)
    mass = float(quanta[quanta > 0].sum())
    state = State.from_host(quanta, np.zeros(quanta.shape))
    diagonal = np.rint(DISTANCES / np.sqrt(3.0)).astype(int)
    axis_arrival = np.full(len(DISTANCES), np.nan)
    diagonal_arrival = np.full(len(DISTANCES), np.nan)
    for count in range(1, 260):
        pull_wave(state, settings)
        pull = state.pull.numpy()
        time = count * settings.step_time
        for index, distance in enumerate(DISTANCES):
            if np.isnan(axis_arrival[index]) and pull[CENTRE_P4 + distance, CENTRE_P4, CENTRE_P4] <= -0.5 * settings.pull_strength * mass / distance:
                axis_arrival[index] = time
            corner = CENTRE_P4 + diagonal[index]
            reach = np.sqrt(3.0) * diagonal[index]
            if np.isnan(diagonal_arrival[index]) and pull[corner, corner, corner] <= -0.5 * settings.pull_strength * mass / reach:
                diagonal_arrival[index] = time
    speed = np.polyfit(axis_arrival, DISTANCES, 1)[0]
    diagonal_speed = np.polyfit(diagonal_arrival, np.sqrt(3.0) * diagonal, 1)[0]
    print(f"axis arrivals {axis_arrival.tolist()} for r {DISTANCES.tolist()}: speed {speed:.4f} "
          f"(set {settings.light_speed}); body diagonal speed {diagonal_speed:.4f}")
    assert not np.isnan(axis_arrival).any()
    assert np.all(np.diff(axis_arrival) > 0)
    assert abs(speed / settings.light_speed - 1.0) <= 0.10


SIDE_P5 = 96


CENTRE_P5 = 48


DIRECTIONS = {"axis": (1, 0, 0), "face_diagonal": (1, 1, 0), "body_diagonal": (1, 1, 1)}


@needs_gpu
def test_the_pull_matches_minus_g_m_over_r_along_axes_and_diagonals():
    settings = Settings()
    places = np.arange(SIDE_P5) - CENTRE_P5
    radius = np.sqrt(places[:, None, None] ** 2 + places[None, :, None] ** 2 + places[None, None, :] ** 2)
    quanta = np.where(radius <= 2.0, 1000, -1000).astype(np.int32)
    mass = float(quanta[quanta > 0].sum())
    state = State.from_host(quanta, np.zeros(quanta.shape))
    for _ in range(160):
        pull_wave(state, settings)
    pull = state.pull.numpy()
    means = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction)
        ratios = []
        for multiple in range(1, 20):
            offset = multiple * unit
            distance = float(np.linalg.norm(offset))
            if 8.0 <= distance <= 16.0:
                there = tuple(CENTRE_P5 + offset)
                ratios.append(pull[there] / (-settings.pull_strength * mass / distance))
        means[name] = float(np.mean(ratios))
        print(f"{name}: pull / (-G M / r) at {len(ratios)} points {np.round(ratios, 4).tolist()}, mean {means[name]:.4f}")
        assert np.all(np.abs(np.asarray(ratios) - 1.0) <= 0.05)
    spread = (max(means.values()) - min(means.values())) / np.mean(list(means.values()))
    print(f"spread of the three means: {spread:.4f}")
    assert spread <= 0.05


@needs_gpu
def test_the_swing_pushes_back_as_steeply_as_the_setting_says():
    import dataclasses
    for steepness in (3.5, 8.25):
        settings = Settings(push_steepness=steepness)
        other = dataclasses.replace(settings, push_steepness=6.0)
        state = new_universe(8, 10, seed=1)
        for _ in range(20):
            step(state, settings)
        load = cell_count(state)
        cell_laws(state, settings)
        joints = (state.kind[0] == -1) & (state.wanted[0] > 0.0)
        before = swing_energy(state, settings, load, joints)
        before_other = swing_energy(state, other, load, joints)
        assert np.abs(state.speed[0, joints]).max() > 1e-3
        pressed = joint_swing(state, settings, load)
        change = abs(swing_energy(state, settings, load, joints) - before) / abs(before)
        change_other = abs(swing_energy(state, other, load, joints) - before_other) / abs(before_other)
        print(f"push_steepness {steepness}: swing energy changes by {change:.2e} of itself; written with steepness 6 "
              f"it changes by {change_other:.2e}; pressed {int(pressed.sum())}")
        assert not pressed.any()
        assert change <= 1e-9
        assert change_other > 1e-6
