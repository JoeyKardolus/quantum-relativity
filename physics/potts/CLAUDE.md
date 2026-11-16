# physics/potts/

The cell engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): fixed
Planck boxes on the GPU (Warp), each owned by one cell or empty; energy volumes (A) and joints
(B) are groups of boxes, a cell's size is its box count (cellular Potts, 1992).

## Must not get wrong

- **The energy fluid is quantum** (owner, 2026-11-16): one complex number per box,
  `universe.psi` (vec2d, float64); energy |psi|^2, flow the phase's turning,
  whirls in whole turns. Gross-Pitaevskii, hbar = box mass = 1.
- **Every update is an exact rotation**: two-box hops (`hop.py`, unitary 2x2, De Raedt's
  product formula) and on-site phase turns (`on_site.py`), Strang-split (`energy_flow.py`).
  Never an explicit Euler step: it breaks the norm. Sum |psi|^2 holds to rounding;
  `fluid_energy` is reported, not conserved.
- **Box moves keep norm, phase continuity and flow**: a given-away box is shared over its
  old cell's face neighbours (complex sums, norm kept); a claimed box and its source each
  hold |source| / sqrt 2, the claimed box's phase continuing the source's flow.
- **Wrapping** (owner, 2026-11-16): no A box has another A's box, or an empty box, among its
  26 neighbours; `box_moves.py` checks all 26 for every box that becomes A. Joints touch
  joints freely (the web). Start: A at all-odd cell
  coordinates off the outer layer, the rest joints, every cell k^3 boxes.
- **Seams** are joint boxes with another joint among their 26 neighbours (`seam_box.py`);
  the fluid crosses into or between joints only there, by the most squeezed joint's openness.
  Friction only on flow crossing an A-B face (as far as the faces behind it carry it: a
  fluid at rest makes no heat); exact, never negative. `step` books other phases as work.
- **Pull** is sourced and felt by A only (third law); its energy is V n / 2.
- **Collapse** merges every A touching the joint at a face, edge or corner (`touch_pairs.py`);
  a joint touching empty space does not collapse.
- **64 phases, blocks 4 apart**: a site's plate, its 26-neighbour check, home and source box
  stay in its own 4x4x4 block. Moves claim only inside the active region.
- **Box kernels run on the active region** (`active_region.py`, PAD 4).
- **Local clocks** (`time_levels.py`): each cell its own level; a face hops at the finer of
  its two boxes' levels, for sqrt(clock_a clock_b) of proper time.
- **Size energy is ln(n / aim)^2**, plus the Potts border term (`surface`).
- **Not bit-repeatable** (move races, float atomics). Never tune.

## Measured (2026-11-16 round 2, seed 1, 8^3 cells of 6^3, defaults)

300 steps, 25 s: 0 lumps, sealed or collapses; occupied x1.69; drift 4e-15; wraps hold.
Heat 42k, pull -17k, compression +64k, moves +394k; balance -3.0% (xfail): splitting
error, moves stir the fluid. Flow carried 8.2 / 7.1 / -2.4 (axis / face / body), rest
scatter 0.03; spin loses its winding (strength 0.86 to -0.01, tilt 90, into the joint).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/potts      # skips without a CUDA GPU
```
