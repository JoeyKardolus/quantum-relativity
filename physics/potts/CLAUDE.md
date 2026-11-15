# physics/potts/

The cell engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): fixed
Planck boxes on the GPU (Warp), each owned by one cell or empty; energy volumes (A) and joints
(B) are groups of boxes, a cell's size is its box count (cellular Potts, Graner and Glazier
1992). Named `potts`: `physics/cells` is an older engine in git history.

## Must not get wrong

- **Energy is whole quanta** (int64, `universe.QUANTUM`). Flow fluxes are one integer
  both boxes compute (`flow.py`); a box move hands its energy on with int64 atomics
  (`box_moves.py`); a collapse only relabels boxes. `stats()["energy_drift"]` is 0.
- **ABAB and neighbours never change** except by joint collapse: a move may not make a new
  contact or end one (contact table, counts changed atomically, undone whole on failure), nor touch like with like, nor split its
  cell (3x3x3 local test). Rebuild the table from boxes (`contact_table.py`) after any relabel.
- **Eight parity phases of moves**: movers sit two boxes apart; at most two share a face
  neighbour, so a claimed box takes a *third* of its claimer's energy, never a half.
- **Kind follows owner.** Pressure is each kind's energy against its own rest (`kick.py`); raw
  energy pressure flooded the joints (measured).
- **Closed faces stop flow** (`kick.py`): momentum into an empty box, the outside or a sealed
  joint border is zeroed; without it momentum piled up at walls (measured).
- **Friction only at A-B crossings** (`flow.py`, keep 1 / (1 + friction c)); none inside a cell.
- **Pull** (`pull.py`, `multigrid.py`) and the start's ripples (`uneven_field.py`) came from the
  removed fluid engine unchanged; the 1/r and slope tests came with them.
- **Wanted size** is `labels * k**3 * (labels * rest / E)**size_power`; labels add on collapse.
- **Not bit-repeatable** (move races, float atomics). **Not built**: volume split, heat field.
- **Honesty.** Never tune for a result; report misses.

## Measured (2026-11-15, RTX 5080)

20^3 cells of 6^3: 1.4 ms per substep; 81 to 366 substeps per step early, 2,000 to 8,000
from step 11 as the pull drives the fluid in. `max_substeps` is 1e6, a runaway guard (2000 stopped default runs). 10^3 cells, 300 steps:
occupied x1.55, flat at the grid's edge.
Expansion check, 8^3 cells of 6^3 (grid 64, start radius 24.0), seed 1, 300 steps, occupied
boxes over the start:

| pull | step 50 | 300 | edge share |
|---|---|---|---|
| 0.005 | 1.53 | 1.57 | 0.012 |
| 0.002 | 1.21 | 1.27 | 0.004 |
| 0.001 | 1.19 | 1.26 | 0.007 |
| 0.0005 | 1.17 | 1.24 | 0.010 |
| 0.0002 | 1.17 | 1.28 | 0.020 |
| 0 | 1.17 | 1.28 | 0.024 |

Space grows at every pull, fastest at the default, slowing as the cells near the walls
(margin 8 boxes); no collapse, no lump, drift 0, ABAB breaks 0 in every run.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/potts      # skips without a CUDA GPU
```
