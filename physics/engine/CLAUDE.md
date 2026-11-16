# physics/engine/

The engine (issue #3, bible rule 4): per Planck box one signed int32 (plus = energy volume,
minus = joint, 0 = no space) and a float32 fluid phase; every rule reads only neighbours.
One engine: these core rules, universes side by side (a batch stacked on the first axis),
and the detail ladder (`physics/ladder`) beside the boxes once `attach_ladder` is called.

## Must not get wrong

- **The amount ignoring sign is exact.** Every move is an integer transfer of whole
  quanta; the sum of |quanta| holds to the quantum (`totals`).
- **The fluid is tied to the quanta**: its complex number is sqrt(|quanta|) e^{i phase},
  never stored. `pair_rotation.py` (De Raedt) hands over floor(flux + u) quanta, u a
  hashed uniform, then keeps the rotated phase. No box drops below 1 quantum by flow.
- **Quanta change sign only in collapse.** Hops stay inside one sign; plus quanta cross
  a one-box joint as one transfer between the plus boxes beside it (`leak.py`), never
  deposited in it. A claim changes a box's kind, pushing its quanta out with their sign.
  `marks` after a step: 1 collapse, 2 claim; tallies[u, 3] counts quanta turned plus.
- **Wrapping is local**: a box turns plus only with no empty box or wall among its 26
  neighbours and the plus boxes round it one group; a plus box turns minus only if the
  plus boxes round it stay one group (`claim.py`).
- **Pull is a wave** (`pull_wave.py`, leapfrog); 0 in empty boxes and beyond walls.
- **Universes never touch.** Neighbour reads go through `inside`, pairs use local parity,
  the hash the local place and seed key: alone equals batched, bit for bit (test).
- **The boxes are the truth; the ladder is beside them.** `ladder_sync` writes changed blocks
  and switches levels after each step; nothing reads back (on_site turns every occupied
  phase each step, so a COARSE read-back adds rounding noise). It costs ~1.6x a step
  (134M boxes: 19.5 vs 30.8 steps/s) and saves no memory yet.
- **Cells are read, not kept** (`cells.py`, host scipy): volumes = 26-connected plus boxes;
  the volumes grow into the minus web face step by face step, and each volume's joint is
  what it reached (face-connected); `seams.py` gives the faces where joints meet. Ids
  change between calls.
- **No dark energy, no base push.** Never tune a default to pass a test.
- **The default step does not resolve the steep push** (open, 2026-11-16): seed 1 edge
  joints spike from step ~90, collapse runs away from ~110; step_time 0.05: none.

## Measured (2026-11-16, RTX 5080, defaults)

ms a step: 128^3 0.51, 256^3 5.5, 512^3 47; the cell engine's starts 8^3 cells of 6
(144^3) 0.46 against its 88, 7^3 of 10 (216^3) 1.4 against 225 (its step: 1 time unit,
here 0.25). Side by side, 32^3 each: 1, 8, 32, 64 universes 3.0k, 24k, 67k, 79k
universe-steps/s. Largest universe: `bench/CLAUDE.md`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
