"""Axiom 09: energy crosses a joint by simple diffusion; friction only on crossing into a joint; friction becomes heat kept as energy.

Audit answers (owner, 2026-11-16: energy crosses a joint by simple diffusion for now) and
bible rule 4 (friction only in passing through a joint; friction destroys no energy, it
turns directed flow into jiggling, which is heat). Staged, Settings() unless named:
  1. diffusion: new_universe(5, 10, seed=1, unevenness=0, margin=8) (8 energy volumes,
     every box 1000) with energy volume V (lattice 2, 2, 2) packed to 2000 per box, and a
     twin at 3000 per box. Simple diffusion (Fick): V's quanta fall at every one of 100
     steps and never below the mean of the other volumes (no overshoot, no slosh back),
     and V's loss over the first 5 steps is proportional to its excess: the twin with
     twice the excess loses 2 times as much, within 1.6 .. 2.4
  2. no friction inside a cell: an 18-box cube of plus boxes (one energy volume, no
     joints; 18**3 boxes stay below split_across**3, so it does not tear) with a directed flow (phase 0.3 x) steps 20 times with the joint crossing
     on (Settings()) and off (crossing_rate 0): quanta and phases are identical (the
     crossing is the only friction, owner 2026-11-16, so the engine has no other
     friction setting; before 2026-11-16 this compared a separate `friction` setting)
  3. heat kept: the 8-volume start (every box 1000) with a directed flow (phase 0.3 x)
     steps 20 times with the crossing on (Settings()) and off (crossing_rate 0, a
     side-by-side comparison): the fluid's kinetic energy, 1/2 sum over same-sign faces of
     |psi_a - psi_b|**2 with psi = sqrt(|quanta|) e^{i phase} (hop.py's kinetic term),
     directed and jiggling together, is with the crossing at least 95% of without
     (heat kept, not deleted); the amount ignoring sign is exact in both
"""
import numpy as np

from physics.engine import Settings, State, cells, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu

V = (2 * 7 + 2) * 7 + 2


def _start(packing: int | None, wave: float) -> State:
    """The 8-volume even start, V at `packing` per box when given, phase wave * x."""
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    if packing is not None:
        quanta[boxes["owner"] == V] = packing
    phase = wave * np.arange(quanta.shape[0], dtype=np.float32)[:, None, None] * np.ones(quanta.shape, np.float32)
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    return State.from_host(quanta, phase, seed=1, lattice=graph)


def _volume_amounts(state: State) -> tuple:
    """(V's quanta, the mean quanta of the other energy volumes)."""
    found = cells(host_boxes(state))
    count = found["volumes"]
    places = found["place"][:count]
    mine = int(np.flatnonzero((places == [2, 2, 2]).all(axis=1))[0])
    others = np.delete(found["quanta"][:count], mine)
    return int(found["quanta"][mine]), float(others.mean())


def _diffusion(packing: int, steps: int) -> list:
    """V's quanta and the others' mean before and after each of `steps` steps."""
    state = _start(packing, 0.0)
    rows = [_volume_amounts(state)]
    for _ in range(steps):
        step(state, Settings())
        rows.append(_volume_amounts(state))
    return rows


def _kinetic(quanta: np.ndarray, phase: np.ndarray) -> float:
    """1/2 sum over faces between two boxes of one sign of |psi_a - psi_b|**2."""
    psi = np.sqrt(np.abs(quanta).astype(np.float64)) * np.exp(1j * phase.astype(np.float64))
    sign = np.sign(quanta)
    total = 0.0
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        same = (sign[tuple(low)] == sign[tuple(high)]) & (sign[tuple(low)] != 0)
        total += 0.5 * float((np.abs(psi[tuple(low)] - psi[tuple(high)]) ** 2)[same].sum())
    return total


@needs_gpu
def test_energy_diffuses_across_joints_without_overshoot_in_proportion_to_the_excess():
    rows = _diffusion(2000, 100)
    twin = _diffusion(3000, 5)
    amounts = np.array([row[0] for row in rows])
    others = np.array([row[1] for row in rows])
    falls = np.diff(amounts)
    ratio = (twin[0][0] - twin[-1][0]) / max(rows[0][0] - rows[5][0], 1)
    rises = int((falls >= 0).sum())
    print(f"V {amounts[0]} -> {amounts[-1]} quanta in 100 steps, others' mean {others[0]:.0f} -> {others[-1]:.0f}; "
          f"steps where V did not fall {rises} of 100; V below the others' mean at {int((amounts < others).sum())} "
          f"steps; loss over 5 steps: excess x1 {rows[0][0] - rows[5][0]}, x2 {twin[0][0] - twin[-1][0]}, "
          f"ratio {ratio:.2f}")
    assert rises == 0 and np.all(amounts >= others)
    assert 1.6 <= ratio <= 2.4


@needs_gpu
def test_no_friction_inside_a_cell():
    quanta = np.full((18,) * 3, 1000, dtype=np.int32)
    phase = 0.3 * np.arange(18, dtype=np.float32)[:, None, None] * np.ones(quanta.shape, np.float32)
    results = []
    for rate in (Settings().crossing_rate, 0.0):
        state = State.from_host(quanta, phase, seed=1)
        for _ in range(20):
            step(state, Settings(crossing_rate=rate))
        results.append((state.quanta.numpy(), state.phase.numpy()))
    same = np.array_equal(results[0][0], results[1][0]) and np.array_equal(results[0][1], results[1][1])
    print(f"one plus cube, 20 steps, crossing on vs off: identical {same}; largest phase gap "
          f"{float(np.abs(results[0][1] - results[1][1]).max()):.3g}")
    assert same


@needs_gpu
def test_friction_turns_flow_into_heat_and_deletes_no_energy():
    kinetic, kept = [], []
    for rate in (Settings().crossing_rate, 0.0):
        state = _start(None, 0.3)
        start = totals(state)["amount"]
        for _ in range(20):
            step(state, Settings(crossing_rate=rate))
        boxes = host_boxes(state)
        kinetic.append(_kinetic(boxes["quanta"], boxes["phase"]))
        kept.append(totals(state)["amount"] - start)
    share = kinetic[0] / kinetic[1]
    print(f"fluid kinetic energy after 20 steps: crossing on {kinetic[0]:.6g}, crossing off "
          f"{kinetic[1]:.6g}, share kept {share:.4f} (need >= 0.95); amount drift {kept}")
    assert kept == [0, 0]
    assert share >= 0.95
