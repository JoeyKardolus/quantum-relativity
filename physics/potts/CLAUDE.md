# physics/potts/

The cell engine (bible `physics/quantum-relativity`, "How the engine holds the foam"): fixed
Planck boxes on the GPU (Warp), each owned by one cell or empty (cellular Potts, 1992).

## Must not get wrong

- **The energy fluid is quantum** (owner, 2026-11-16): one complex number per box,
  `universe.psi` (vec2d, float64); energy |psi|^2, flow the phase's turning,
  whirls in whole turns; hbar = box mass = 1.
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
  fluid at rest makes no heat); never negative. Its heat (a tally) is spread as energy
  over the pair and its face neighbours in both cells, a joint's only at seams
  (`spread_heat.py`, rule 8 drift); `step` books other phases as work.
- **Pull** is sourced and felt by A only (third law); its energy is V n / 2.
- **Collapse** merges every A touching the joint at a face, edge or corner (`touch_pairs.py`);
  a joint touching empty space does not collapse.
- **64 phases, blocks 4 apart**: a site's plate, its 26-neighbour check, home and source box
  stay in its own 4x4x4 block. Moves claim only inside the active region. Each sweep shuffles
  the phase order (`phase_order`): a fixed x-major order made x, y, z carry flow unequally.
- **Box kernels run on the active region** (`active_region.py`, PAD 4).
- **Local clocks** (`time_levels.py`): each cell its own level; a face hops at the finer of
  its two boxes' levels, for sqrt(clock_a clock_b) of proper time. `cell_time` is each
  cell's reading (`local_time.py`).
- **Size energy is ln(n / aim)^2**, plus the Potts border term (`surface`).
- **Not bit-repeatable** (move races, atomics). Never tune.

## Measured (2026-11-16, seed 1, defaults: 7^3 cells of 10^3)

300 steps, 236 ms a step: occupied x1.63, growth 1.46% a step falling to 0.005%,
0.019% at the end; 21 lumps, none sealed, no collapse; drift 1e-14.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/potts      # skips without a CUDA GPU
```
