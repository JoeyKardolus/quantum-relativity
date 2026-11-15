# physics/potts/

The cell engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): fixed
Planck boxes on the GPU (Warp), each owned by one cell or empty; energy volumes (A) and joints
(B) are groups of boxes, a cell's size is its box count (cellular Potts, Graner and Glazier
1992). Named `potts`: `physics/cells` is an older engine in git history.

## Must not get wrong

- **Energy is whole quanta** (int64, `universe.QUANTUM`). Flow fluxes are one integer both
  boxes compute (`flow.py`); plate moves hand energy on inside their own block; collapse and
  split only relabel. `stats()["energy_drift"]` is 0. Heat is bookkeeping, not energy.
- **A joint is a full cell** (rule 8): same energy per box at the start, one size law and one
  `cell_rest` for both kinds, one sound speed. No thin-wall share anywhere.
- **ABAB and neighbours never change** except by collapse or split. Plate moves (`box_moves.py`)
  check every face of every changed box, net face counts per pair atomically (undone whole),
  and connectivity in the 4x4x3 block. One-box moves froze edges and corners; never go back.
- **64 phases, blocks 4 apart**: a site's plate, ring, deposit and source box stay in its own
  4x4x4 block, so a claimed box may take *half* its neighbour's energy. Moves claim only
  inside the active region.
- **Box kernels run on the active region** (`active_region.py`, PAD 4); pull solves on a cube
  round it. The whole grid at 24^3 cells is 80M boxes.
- **Local clocks** (`time_levels.py`): each cell its own level; boxes ordered by level so the
  due ones are a prefix. Flow takes in from finer neighbours via the flow order.
- **Pressure in ln e** (`kick.py`); closed and half-open faces stop the flow reaching them
  (heat) and push back on the force; grad P flung few-quanta boxes at 1e9 boxes/time.
- **Size energy is ln(n / aim)^2**, plus the Potts border term (`surface`): without it cells
  grew into sponges and tears failed.
- **Seams**: energy enters or leaves a joint only at seam boxes (`seam_flags.py`).
- **Not bit-repeatable** (move races, float atomics). Never tune for a result.

## Measured (2026-11-15, RTX 5080, seed 1, k = 6, defaults)

Old substep climb: momentum piled up against nearly sealed seams in thin boxes; one global
substep served the fastest box.
Speed, 30 steps after the first: 8^3 cells 49 steps/s; 16^3 4.4 (2.6 by step 31, space
x3.3); 24^3 1.7 (1.1 by step 31). Mostly 32 finest substeps per step.
Run, 16^3 cells, 300 steps (525 s): occupied x1.25 (correction), x2.47 at 25, x8.4 at 50,
x24.8 at 300; the region reached the grid's walls by step 55 (edge share 1e-4 after).
5,019 splits, 4,991 collapses (new tears drain and collapse), 2,069 tears refused; lumps
144 -> 244 (step 25) -> 6; sealed lumps 0; drift 0, ABAB breaks 0 throughout.
Pull 0 (8^3, 30 steps): x1.40, pull on x2.24: growth comes from lumps.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/potts      # skips without a CUDA GPU
```
