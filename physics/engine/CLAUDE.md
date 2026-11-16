# physics/engine/

The engine (issue #3, bible rule 4): per Planck box one signed int32 (plus = energy volume,
minus = joint, 0 = no space) and a float32 fluid phase. Universes run side by side; the
detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **The amount ignoring sign is exact.** Every move is an integer transfer of whole
  quanta; the sum of |quanta| holds to the quantum (`totals`).
- **The fluid is tied to the quanta**: sqrt(|quanta|) e^{i phase}, never stored;
  `pair_rotation.py` hands over whole quanta. No box drops below 1 quantum by flow.
- **Quanta change sign only in collapse.** Hops stay inside one sign; plus quanta cross
  one joint of any thickness as one transfer between the plus boxes at the ends of a
  straight plus-minus...-plus line (`leak.py`), never deposited in it. A claim changes a
  box's kind after pushing its quanta out to boxes of its old kind (owner, 2026-11-16).
  `marks` after a step: 1 collapse, 2 claim; tallies[u, 3] counts quanta turned plus.
- **Wrapping is local**: a box turns plus only with no empty box or wall among its 26
  neighbours and the plus boxes round it one group; a plus box turns minus only if the
  plus boxes round it stay one group, and a minus box turns plus only if the minus boxes
  in its 18-neighbourhood stay one face-joined group, so the web never breaks (`claim.py`).
- **Pull is a wave** (`pull_wave.py`); 0 in empty boxes and beyond walls.
- **Universes never touch.** Reads go through `inside`; substeps are per universe:
  alone equals batched, bit for bit (test).
- **The boxes are the truth; the ladder is beside them.** `ladder_sync` writes changed
  blocks after each step; nothing reads back (a COARSE read-back adds rounding noise).
- **Cells are read, not kept** (`cells.py`, host scipy): volumes = 26-connected plus boxes;
  volumes and empty space grow into the minus web face step by face step, each minus box
  keeps its first two sources, and a joint is every box of one source pair: exactly one
  shared joint between two volumes (owner, 2026-11-16). Ids change between calls. The leak cannot read joints, so after 300
  steps only ~2/3 of its crossings run through exactly their own joint (open).
- **No dark energy, no base push.** Never tune a default to pass a test.
- **Substeps resolve the steep push** (`substeps.py`, numerics): each universe cuts its
  fluid step so no same-sign face turns more than max_turn (0.25 rad) a substep. Seed 1
  default start: coarse step (max_turn 0) squeeze 4.1 at step 100, collapse from ~115;
  substeps (~4 a step): squeeze <= 0.59, no collapse in 300 steps.

## Measured (2026-11-16, RTX 5080, defaults)

Before substeps, ms a step: 128^3 0.51, 256^3 5.5, 512^3 47; with them ~4x more on the
seed 1 default start (~4 substeps a step). Largest universe: `bench/CLAUDE.md`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
