# physics/potts/

The cell engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): fixed
Planck boxes on the GPU (Warp), each owned by one cell or empty; energy volumes (A) and joints
(B) are groups of boxes, a cell's size is its box count (cellular Potts, Graner and Glazier
1992). Named `potts`: `physics/cells` is an older engine in git history.

## Must not get wrong

- **The energy fluid is quantum** (owner, 2026-11-16): one complex number per box,
  `universe.psi` (vec2d, float64); a box's energy is |psi|^2, its flow is the phase turning
  from box to box, whirls turn in whole steps. Gross-Pitaevskii, hbar = box mass = 1.
- **Every update is an exact rotation**: two-box hops (`hop.py`, unitary 2x2, De Raedt's
  product formula) and on-site phase turns (`on_site.py`), Strang-split (`energy_flow.py`).
  Never an explicit Euler step: it breaks the norm. Total energy (sum |psi|^2) holds to
  float64 rounding (`stats()["relative_drift"]`); `fluid_energy` is reported, not conserved.
- **Box moves keep the norm**: a box given away grows its home box to hold both energies
  (home's phase kept); a claimed box and its source each hold source psi / sqrt 2.
- **Wrapping** (owner, 2026-11-16): no A box has another A's box, or an empty box, among its
  26 neighbours; `box_moves.py` checks all 26 for every box that becomes A. Joints touch
  joints freely (the web); which joints touch which may change. Start: A at all-odd cell
  coordinates off the outer layer, the rest joints, every cell k^3 boxes.
- **Seams** are joint boxes with another joint among their 26 neighbours (`seam_box.py`);
  the fluid crosses into or between joints only there, by the most squeezed joint's openness.
  Friction only at A-B faces; its heat is the clocked flow-energy drop on every face the
  turned boxes touch; `step` books other phases as work (`stats()["energy_balance"]`).
- **Pull** is V in every box's phase turn (no step at A-B faces); sourced by A only.
- **Collapse** merges every A touching the joint at a face, edge or corner (`touch_pairs.py`);
  a joint touching empty space does not collapse.
- **64 phases, blocks 4 apart**: a site's plate, its 26-neighbour check, home and source box
  stay in its own 4x4x4 block. Moves claim only inside the active region.
- **Box kernels run on the active region** (`active_region.py`, PAD 4).
- **Local clocks** (`time_levels.py`): each cell its own level; a face hops at the finer of
  its two boxes' levels, for sqrt(clock_a clock_b) of proper time.
- **Size energy is ln(n / aim)^2**, plus the Potts border term (`surface`).
- **Not bit-repeatable** (move races, float atomics). Never tune for a result.

## Measured (2026-11-16, RTX 5080, seed 1, 8^3 cells of 6^3 boxes, defaults)

27 volumes, 485 joints; 300 steps in 14 s. Relative drift 4e-15; wrap breaks 0, web one
piece; 0-1 collapses; occupied x1.72, edge share 0. Heat 619k, work pull -165k, compression
+276k, moves +716k; balance 75,324 to 72,022 (-4.4%).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/potts      # skips without a CUDA GPU
```
