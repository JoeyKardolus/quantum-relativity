# physics/fluid/

The GPU engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): energy
as a fluid on a fixed cube of boxes, Warp kernels on CUDA. Every box holds A (pulls) and B
(pushes) energy and a momentum per kind. Module map in `__init__.py`.

## Must not get wrong

- **Energy is whole quanta** (`grid.py`, int64, `QUANTUM` = 2^-32). Every face flux is one
  integer taken from one box and given to the other, so `stats()["energy_drift"]` is 0, not
  small. Never store energy as float or clamp a box: either breaks exactness.
- **Donor cell, not Rusanov** (`transport.py`): a box sends by its own velocity, so the energy
  moved equals the momentum held and a blob's centre moves at P / M. Rusanov's wave-speed
  diffusion slowed a moving collapsing blob 32% (measured); do not switch back.
- **Pull acts on A only**, sourced by A only (`pull_kick.py`). Pressure kicks B only
  (`push_kick.py`, face pressure = the smaller side, so an empty box is never kicked).
  A meets B only in `rub.py`.
- **Seal is in B's frame**: `rub.py` locks A to B in a sealed box and its face neighbours,
  before transport. Zero A flux in the grid's frame pins a moving sealed lump; do not.
- **Friction only in `rub.py`** (rate friction x compression). No viscosity anywhere else.
- **Order per substep** (`step.py`): potential, CFL dt, pull kick, push kick, rub, transport.
- **The substep is set by content, not vapour** (`fastest_wave.py`): a kind under `cfl_floor`
  in a box, and flow into a closed wall (`sending_speed.py`), set nothing; such vapour sends at
  most its content (`transport.py`). Counting them took 3 to 10 times the substeps.
- **The start** (`new_grid.py`, `uneven_field.py`): log energy is a Gaussian field, P(k) =
  k**slope (default -3, every size alike), rms `unevenness`; B = 3% of A in every box.
- **Walls are closed**; `stats()["edge_share"]` says when energy has reached them. A run whose
  edge share is not ~0 is no longer open space; report it.
- **Honesty.** Never tune a setting until a result passes; report misses with seed and
  settings. Today (seed 1, unevenness 0.3, defaults) the whole filled cube falls together
  into one core cluster by step ~10 (free fall 7.7 time units at density 1, any size). Drift 0;
  runs repeat exactly (no float atomics).

## Measured (2026-11-15, RTX 5080, filled start, seed 1, unevenness 0.3)

ms per substep 2.7 at side 128, 18 to 21 at 256. Substeps per step at 128: 11 to 61 (steps
1 to 6), up to 377 in the fall, 240 by step 45; at 256: 15 to 110, up to 1072, then 580 to
970; 200 steps at 256 took 48 min. Once fallen the substep is set by real matter at 95 to 150
boxes per time unit near the core. GPU memory 0.35 GiB at 128, 2.75 GiB at 256.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/fluid      # skips without a CUDA GPU
```
