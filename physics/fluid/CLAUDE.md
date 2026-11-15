# physics/fluid/

The GPU engine (bible `physics/quantum-relativity`): energy as a fluid on fixed boxes, Warp
on CUDA; A (pulls) and B (pushes) energy and a momentum per kind. Module map in `__init__.py`.

## Must not get wrong

- **Energy is whole quanta** (`grid.py`, int64, `QUANTUM` = 2^-32). Every face flux is one
  integer taken from one box and given to the other: `stats()["energy_drift"]` is 0. Never
  store energy as float or clamp a box.
- **Donor cell, not Rusanov** (`transport.py`): a box sends by its own velocity, so energy
  moved equals momentum held. Rusanov slowed a moving blob 32% (measured); do not switch back.
- **Pull acts on A only**, sourced by A only. Pressure kicks B only (face pressure = the
  smaller side). A meets B in `rub.py` and at faces (`a_passage.py`).
- **Seal is in B's frame**: `rub.py` locks A to B in a sealed box and its face neighbours,
  before transport. Zero A flux in the grid's frame pins a moving sealed lump; do not.
- **Friction only in `rub.py`** (rate friction x compression). No viscosity anywhere else.
  Its loss goes to `grid.heat` (float, apart from the exact quanta), carried with energy.
- **Local time** (`clock_rate.py`): every process in a box runs for clock * dt, CFL too.
  Uneven clocks break the pull's action = reaction in counter time; test momentum with even clocks.
- **A moves only through joints**: `a_passage.py` in transport, `leak.py` (by pull, shut
  at the seal). Lengths are measured (`local_length.py`), never fed back into the dynamics.
- **Order per substep** (`step.py`): potential, clocks, dt, pull kick, push kick, rub,
  leak, transport.
- **The substep is set by content, not vapour** (`fastest_wave.py`): a kind under `cfl_floor`
  and flow into a closed wall set nothing; such vapour sends at most its content.
- **The start** (`new_grid.py`): log energy a Gaussian field, P(k) = k**slope (default -3),
  rms `unevenness`; B = 3% of A in every box.
- **Walls are closed**; a run whose `edge_share` is not ~0 is no longer open space; report it.
- **Honesty.** Never tune a setting until a result passes; report misses with seed and
  settings. Today (seed 1, unevenness 0.3, defaults) the filled cube still falls together
  into one lump by step 10; slow clocks stretch the core's collapse (max compression 1583 at
  step 40, was 5e5 by step 10). Drift 0; runs repeat exactly.

## Measured (2026-11-15, RTX 5080, side 128, seed 1, unevenness 0.3)

Substeps per step 8 to 75 (steps 1 to 5), 246 at 20, 786 at 40 (before local clocks: 11
to 45, 292, 239); ms per step 2812 at 40 (was 613), 3.6 ms per substep. The substep is now
set by A of 0.001 per box with no B falling into the core at 50 to 74 boxes per time unit,
clock 2, not by the core. `expansion` is dominated by thin boxes (length 1 / energy).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/fluid      # skips without a CUDA GPU
```
