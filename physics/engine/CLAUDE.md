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
- **Joints swing, volumes follow.** `joint_swing` moves joint sizes like springs with
  inertia (summed size kept); `exchange` and `mortar_shift` move boxes toward the sizes,
  a box changing cell only once emptied, every volume wrapped on all 26 sides.
- **Local clocks**: every per-cell rule runs at its cell's rate (`clock_rate`).
- **Wrapping**: `new_universe` with margin 0 wraps round when its side is even, else has
  walls. Open-space scenes pass a margin (`physics/scenes/open_margin.py`).
- **Universes never touch** (`reach`); alone equals batched (test).
- **No dark energy, no base push.** Never tune a default to pass a test.

## Measured (2026-11-16, RTX 5080, defaults, seed 1)

Default universe (16 places of 10 boxes, wraps, 88^3, 512 volumes): the 1000-step run
is on issue #1. Axiom tests red at landing:
02 (a staged collapse turns 631 plus boxes minus), 05 (joints reach 34% of the boxes,
at most 33% allowed), 12 (11 of 1,536 joints never swing back), and the 500-step whirl
(one winding flips by step 500).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
