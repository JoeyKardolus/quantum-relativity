# physics/engine/

The engine (bible rule 4): per Planck box one signed int32 (plus = energy volume, minus =
joint, 0 = no space) and a float32 fluid phase, on a cell graph. Universes run side by
side; the detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **One test per axiom** (`tests/test_axiom_NN_*.py`, owner 2026-11-16). Never loosen
  one to pass; a red axiom test is a miss to report, and the engine is what changes.
- **The amount ignoring sign is exact.** Every move is a whole-quanta transfer
  (`totals`). The fluid is sqrt(|quanta|) e^{i phase}, never stored.
- **Sign changes only in collapse (minus to plus) and a split's tear (plus to minus).**
  Hops stay in one cell; `cross` diffuses plus quanta from volume to volume through
  their shared joint, never into it. `marks`: 1 collapse, 2 owner change, 8 split.
- **The cells are a graph** (`state.kind`, `state.between`): a joint knows its two
  volumes, a volume its joints. Collapse merges two volumes into one cell; a split
  touches only the too-big volume. Never read labels off the signs.
- **No size at rest, no drift setting.** A cell wants (size_energy / quanta)**3 boxes
  (`cell_laws`), the constant fixed once by `new_universe`'s correction. Volumes only
  pull, joints only push, empty space does nothing.
- **Joints swing, volumes follow.** `joint_swing` is a frictionless spring step
  (implicit, discrete gradient) keeping swing energy and summed size exactly; an
  explicit step pumped energy in at the steep wall. A joint pressed to the collapse
  limit in the swing collapses that step. Boxes follow sizes by relative room
  (`exchange`, `mortar_shift`, `cell_room`), never sizes boxes; a box changes cell only
  once emptied, every volume wrapped on all 26 sides.
- **Deterministic.** Box moves are proposed, then committed against their phase's worst
  case (`settle_counts`): one seed, one run. No put-back on a racing atomic.
- **Local clocks**: every per-cell rule runs at its cell's rate (`clock_rate`).
- **Wrapping**: `new_universe` with margin 0 wraps round when its side is even, else has
  walls. Open-space scenes pass a margin (`physics/scenes/open_margin.py`).
- **Universes never touch** (`reach`); alone equals batched (test).
- **No dark energy, no base push.** Never tune a default to pass a test.

## Measured (2026-11-16, RTX 5080, defaults, seed 1)

Default universe (16 places of 10 boxes, wraps, 88^3, 512 volumes), 2000 steps at ~12
steps/s: no stall, amount exact, 0 collapses, 0 splits, lumps 82 -> 4, joints' share of
boxes 25% -> 32% (diffusion evens the volumes' energy, so their summed wanted size
shrinks and the closed universe's room goes to every cell by share). Red: axiom 12
(joints swing slowly, median 1 reversal in 300 steps; the old reversals were the
explicit step's energy pump) and the 500-step whirl (the free fluid empties the core;
whole quanta in boxes of 1-20 scramble the small loops).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
