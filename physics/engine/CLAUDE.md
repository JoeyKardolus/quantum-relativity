# physics/engine/

The engine (bible rule 4): per Planck box one signed int32 (plus = energy volume, minus =
joint, 0 = no space) and a float32 fluid phase, on a cell graph. Universes run side by
side; the detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **One test file per owner axiom** (`tests/test_axiom_<n>_*.py`, n 1-8 as in the
  bible's "The axioms"; axiom 7, quantum, is xfail "not built"). Never loosen one to
  pass; a red axiom test is a miss to report, and the engine is what changes.
- **The amount ignoring sign is exact.** Every move is a whole-quanta transfer
  (`totals`). The fluid is sqrt(|quanta|) e^{i phase}, never stored.
- **Sign changes only in collapse (minus to plus) and a split's tear (plus to minus).**
  Hops stay in one cell; `cross` moves plus quanta from volume to volume through
  their shared joint, never into it: drift down the pull plus diffusion (Smoluchowski),
  scaled by the fluid's own jiggle (`jiggle`), no setting.
  `marks`: 1 collapse, 2 owner change, 8 split.
- **The cells are a graph** (`state.kind`, `state.between`): a joint knows its two
  volumes, a volume its joints. Collapse merges two volumes into one cell; a split
  touches only the too-big volume. Never read labels off the signs. A collapse keeps
  minus every joint box that bridges two parts of the web; a split's joint pieces keep
  the whole's squeeze (a piece with less energy wants more room).
- **No size at rest, no drift setting.** A cell wants (size_energy / quanta)**3 boxes
  (`cell_laws`), the constant fixed once by `new_universe`'s correction. Volumes only
  pull, joints only push, empty space does nothing.
- **Joints swing, volumes follow.** `joint_swing` is a frictionless spring step
  (implicit, discrete gradient) keeping swing energy and summed size exactly. A joint pressed to the collapse
  limit in the swing collapses that step. Boxes follow sizes by relative room
  (`exchange`, `mortar_shift`, `cell_room`), never sizes boxes; a box changes cell only
  once emptied, every volume wrapped on all 26 sides.
- **Deterministic.** Box moves are proposed, then committed against their phase's worst
  case (`settle_counts`): one seed, one run. No put-back on a racing atomic.
- **Local clocks**: every per-cell rule runs at its cell's rate (`clock_rate`).
- **Wrapping**: margin 0 wraps when the side is even, else walls (`open_margin.py`).
- **Universes never touch** (`reach`). No dark energy, no base push.

## Measured (2026-11-16, RTX 5080, defaults, seed 1)

Default universe (16 places of 10 boxes, wraps, 88^3, 512 volumes), 2000 steps at ~8.4
steps/s: amount exact, 0 collapses, 0 splits, lumps 165 at step 200 and 173 at 2000
(plain diffusion: 82 -> 4), joints' share of boxes 31% -> 41%, still rising. Strict xfail: axiom
2 swing (joints start at rest at balance, nothing jolts them: median 2 reversals in 300
steps) and the 500-step whirl (one loop reads -1 at step 500).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
