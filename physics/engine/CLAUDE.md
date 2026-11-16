# physics/engine/

The engine (bible rule 4): per Planck box one signed int32 (plus = energy volume, minus =
joint, 0 = no space) and a float32 fluid phase, on a cell graph. Universes run side by
side; the detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **One test file per owner axiom** (`tests/test_axiom_<n>_*.py`, n 1-8 as in the
  bible's "The axioms"). Never loosen one to
  pass; a red axiom test is a miss to report, and the engine is what changes.
- **Axiom 7** (`quantum*.py`): a moving energy is a record in `state.quantum`. Only a
  scene splits it (`quantum_hand`, staged: `cross` strips all direction); a settle
  moves whole quanta. Bell tests: not built.
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
  minus every joint box bridging two parts of the web; a split's joint pieces keep the
  whole's squeeze.
- **No size at rest, no drift setting.** A cell wants (size_energy / quanta)**3 boxes
  (`cell_laws`), the constant fixed once by `new_universe`. Volumes only pull, joints
  only push, empty space does nothing.
- **Collapse only when too packed**: a joint's minus quanta per box
  past `collapse_packing`, never low energy. A tear's joint is born at its wanted size.
- **The fluid has its own pressure** (`fluid_potential`, push * (n / cell mean - 1)):
  whirl cores hold. The volumes' fluid presses joints (`wall_press`), so
  joints swing from the start. Cells start with at least `least_cell_boxes` (216).
- **Joints swing, volumes follow.** `joint_swing`: frictionless, energy-keeping spring
  step. Boxes follow sizes (`exchange`, `mortar_shift`, both by room order, joint
  included); a box changes cell only once emptied, every volume wrapped on 26 sides.
- **Deterministic.** Box moves are proposed, then committed against their phase's worst
  case (`settle_counts`): one seed, one run.
- **Local clocks**: every per-cell rule runs at its cell's rate (`clock_rate`).
- **Wrapping**: margin 0 wraps on an even side. Universes never touch (`reach`).

## Measured (2026-11-16, RTX 5080, defaults, seed 1)

Default universe (88^3, 512 volumes), 2000 steps: amount exact, 0 collapses, 0 splits,
lumps 126 -> 131, joints' share of boxes 25% (start) -> 30%, slowing. Strict xfail: axiom 2 swing (1 of 1536 joints swings
only twice in 300 steps).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
