# physics/engine/

The engine (bible rule 4): per Planck box one signed int32 (plus = energy volume, minus =
joint, 0 = no space) and a float32 fluid phase, on a cell graph. Universes run side by
side; the detail ladder (`physics/ladder`) may sit beside.

## Must not get wrong

- **One test file per owner axiom** (`tests/test_axiom_<n>_*.py`). Never loosen one;
  a red one is a miss to report, and the engine changes.
- **Axiom 7** (`quantum*.py`): a moving energy is a record in `state.quantum`. Only a
  scene splits it (`quantum_hand`, staged: `cross` strips all direction); a settle
  moves whole quanta. Bell tests: not built.
- **The amount ignoring sign is exact** (`totals`): every move is whole quanta. The
  fluid is sqrt(|quanta|) e^{i phase}, never stored.
- **Sign changes only in collapse and a split's tear.**
  Hops stay in one cell; `cross` moves plus quanta from volume to volume through
  their joint, never into it: drift down the pull plus diffusion, no setting.
  `marks`: 1 collapse, 2 owner change, 8 split.
- **The cells are a graph** (`state.kind`, `state.between`): a joint knows its two
  volumes. Collapse merges two volumes into one cell, keeping minus every joint box
  bridging the web; a split touches only the too-big volume. A cut joint's pieces keep
  the whole's squeeze, joined joints their pooled squeeze (mirrors; sizes are never
  added). Never read labels off the signs.
- **No size at rest, no drift setting.** A cell wants (size_energy / quanta)**3 boxes
  (`cell_laws`), the constant fixed once by `new_universe`.
- **Collapse only when too packed**: a joint's minus quanta per box, or per its own
  size, past `collapse_packing`, never low energy; none waits. A tear's joint is
  born at its wanted size.
- **The fluid has its own pressure** (`fluid_potential`, push * (n / cell mean - 1)):
  whirl cores hold. It presses joints (`wall_press`), so they swing from the
  start. Cells start with at least `least_cell_boxes` (216).
- **Joints swing, volumes follow.** `joint_swing`: frictionless, energy-keeping spring
  step on the GPU, retried finer when unsettled. A stopped joint is in no spring and
  hands on its speed (`speed_handover`). Boxes follow sizes
  (`exchange`, `mortar_shift`); a box changes cell only once emptied.
- **No rule copies every box to the host.** `new_universe` builds on the GPU, a tear
  and a collapse read only their room (`cells_span`), each cell takes its own fluid
  substeps (`later_substeps`).
- **Deterministic.** Box moves are proposed, then committed against their phase's worst
  case (`settle_counts`); per-cell sums whole or in fixed order.
- **Local clocks**: every per-cell rule runs at its cell's rate.
- **Wrapping**: margin 0 wraps on an even side. Universes never touch.

## Measured (2026-11-17, RTX 5080, seed 1)

Defaults, 24 places wrapping, 20,000 steps: 1 collapse, 0 tears, nothing sealed
(squeeze at most 0.34, seal 1). Seal 0.1, pull x10: volumes seal singly, never a region,
and reopen; collapses chain. Issue #1.
Strict xfail: axiom 2 swing.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
