"""Axiom 13: everything inside a cell (pull, resizing, flow, collapse) runs on that cell's local clock, slower where more packed.

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
"""
import numpy as np
from scipy.stats import spearmanr

from physics.engine import Settings, new_universe, step
from physics.engine.tests._gpu import needs_gpu

MOST = -0.7


@needs_gpu
def test_more_packed_cells_live_less_time():
    from physics.engine.tests.axiom_run import axiom_run
    found = axiom_run()["checks"][-1]["cells"]
    count = found["volumes"]
    packing = found["quanta"] / np.maximum(found["boxes"], 1)
    volume_rho = spearmanr(packing[:count], found["clock"][:count])[0]
    joint_rho = spearmanr(packing[count:], found["clock"][count:])[0]
    print(f"step 300: local time vs packing, volumes rho {volume_rho:+.3f}, joints rho {joint_rho:+.3f} (need <= {MOST}); "
          f"local time {found['clock'].min():.3f} .. {found['clock'].max():.3f} of counter time 75")
    assert volume_rho <= MOST and joint_rho <= MOST


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
