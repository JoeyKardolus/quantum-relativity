# physics/engine/

The engine (issue #3, bible rule 4): per Planck box one signed int32 (plus = energy volume,
minus = joint, 0 = no space) and a float32 fluid phase; every rule reads only neighbours.
One engine: these core rules, universes side by side (a batch stacked on the first axis),
and the detail ladder (`physics/ladder`) beside the boxes once `attach_ladder` is called.

## Must not get wrong

- **The amount ignoring sign is exact.** Every move of energy is an integer transfer
  between boxes; nothing rounds the quanta. A new rule that moves energy must move whole
  quanta and keep the sum of |quanta| to the quantum (`totals`).
- **The fluid is tied to the quanta**: its complex number is sqrt(|quanta|) e^{i phase},
  never stored. `hop.py` (De Raedt pair rotations) hands over floor(flux + u) quanta,
  u a hashed uniform, then keeps the rotated phase. No box drops below 1 quantum by flow.
- **Signs change only by claims and collapse.** Quanta crossing a plus-minus face join the
  receiving box's kind; a claim turns a border box with whole quanta pushed out to its
  own kind; only `collapse.py` turns minus quanta plus.
- **Wrapping is local**: a box turns plus only with no empty box or wall among its 26
  neighbours and the plus boxes round it one group; a plus box turns minus only if the
  plus boxes round it stay one group (`claim.py`).
- **No joint labels**: a "seam" is any plus-minus face, graded by the joint box's squeeze
  (`openness.py`). The bible's seam (where two joints touch) cannot be read from signs.
- **Pull is a wave** (`pull_wave.py`, leapfrog); 0 in empty boxes and beyond walls.
- **Universes never touch.** Every neighbour read goes through `inside(box, home, side)`,
  hops pair by local parity, the rounding hash uses the local place and the universe's seed
  key: a universe steps bit for bit the same alone and in a batch (test).
- **The boxes are the truth; the ladder is beside them.** `ladder_sync` writes changed blocks
  and switches levels after each step; nothing reads back from it, because on_site turns
  every occupied box's phase every step, so no occupied block is ever unchanged and a COARSE
  block read back would take fresh rounding noise every step. Costs ~13x a step at 216^3.
- **Cells are read, not kept** (`cells.py`, host scipy): volumes = 26-connected plus boxes,
  joints = face-connected minus boxes; ids change between calls.
- **No dark energy, no base push.** Never tune a default to pass a test.

## Measured (2026-11-16, RTX 5080, defaults)

ms a step: 128^3 0.51, 256^3 5.5, 512^3 47; the cell engine's starts 8^3 cells of 6
(144^3) 0.46 against its 88, 7^3 of 10 (216^3) 1.4 against 225 (its step: 1 time unit,
here 0.25). Side by side, 32^3 each: 1, 8, 32, 64 universes 3.0k, 24k, 67k, 79k
universe-steps/s. Largest universe: `bench/CLAUDE.md`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
