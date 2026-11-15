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
- **Walls are closed**; `stats()["edge_share"]` says when energy has reached them. A run whose
  edge share is not ~0 is no longer open space; report it.
- **Honesty.** Never tune a setting until a result passes; report misses with seed and
  settings. Today (seed 1, side 128, unevenness 0.5, defaults): the whole ball falls into one
  box-sized core by step 8 (free fall about 7.7 time units); 47 lumps at step 4, 1 from step 5,
  sealed from step 10; drift 0 quanta. Step 48 needs more than 2000 substeps (`max_substeps`):
  velocities into a one-box core. Runs repeat exactly (no float atomics).

## Measured (2026-11-15, RTX 5080, steps 2 to 21)

ms per step 28 / 296 / 912 at side 64 / 128 / 192 (41 / 120 / 122 substeps of 0.67 / 2.5 /
7.5 ms; potential 0.4 / 1.0 / 2.9 ms as one CUDA graph, transport 0.08 / 0.94 / 3.2 ms).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/fluid      # skips without a CUDA GPU
```
