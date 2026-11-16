# physics/ladder/

The detail ladder (issue #3): blocks of 8^3 Planck boxes held at FULL, COARSE, REUSED or
EMPTY, a library the core engine calls. Each box: one signed whole number of quanta
(int32; plus pulls, minus pushes, 0 no space, bible rule 4) and the fluid's phase (float32).

## Must not get wrong

- **Switching never changes a quantum.** No level move may change a box's sign or the
  plus and minus totals; COARSE keeps each 4^3 group's plus and minus sums exactly (floor
  coding, the dropped rest stored and handed back in rank order, `_kernels.py`). A new
  coding must keep that, and the round-trip and wild-field tests must stay exact.
- **Whirls and flow are never coded.** A block with a whole-turn square or a face phase
  step above `flow_tol` (measured with a one-box halo) is FULL; COARSE and REUSED come
  back to FULL on the pass that sees it.
- **REUSED is exact.** Sharing is keyed by the block's full bytes (quanta and phase),
  never a hash alone; `memo_step` keys include the halo and the rule key, so a block at a
  different neighbourhood or under different settings never takes another's result.
- **Error is stated, not tuned.** COARSE error per box <= step - 1 quanta, allowed only
  when <= max_rel_error (1%) of the group's smallest magnitude; phase <= max_phase_error
  (0.02 rad). These are storage accuracy, never physics settings.
- **Writes go through `write_blocks`**; unchanged blocks are skipped so a quiet block at
  rest gathers no re-coding error. A COARSE block the core rewrites with new content is
  re-coded on the next pass: its rounding noise (<= step - 1 a box, totals exact) comes
  back each time it changes.
- **Lower levels replace FULL rows.** `switch_levels` ends by compacting every pool under
  half used (`_Pool.shrink`: rows moved bit for bit, slots and shared keys renumbered in
  `_moves._compact`; a new holder of slot numbers must be renumbered there too).

## How the core calls it, once per step

```
ids = np.argwhere(store.level != Level.EMPTY)              # or the active region
quanta, phase = gather_blocks(store, ids, halo=REACH)      # any level, decoded
new_q, new_p = core_step(quanta, phase)                    # (n, 8, 8, 8), totals the core's job
write_blocks(store, ids, new_q, new_p)                     # only changed blocks become FULL
switch_levels(store)                                       # measure, promote, share, code
```
Quiet REUSED blocks can step through `memo_step(memo, quanta, phase, core_step)`: one
run per distinct block-with-halo. `promote_blocks` forces a region FULL ahead of an event.

## Measured (2026-11-16, RTX 5080)

Bytes a box: FULL 8, COARSE 1.75, REUSED 4096 / (512 x holders), EMPTY 0 + 0.039 index.
Quiet 134M-box field all COARSE: allocated 9.11 -> 2.01 B a box, GPU held 1.55 -> 0.34
GB, totals exact, first switch 1.42 s. Beside the engine's default start no occupied
block is quiet, so only EMPTY saves (0.35 B a box in use). Bench numbers: `bench/CLAUDE.md`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/ladder                       # Warp CPU device
LADDER_TEST_DEVICE=cuda:0 PYTHONPATH=. .venv/bin/pytest -q physics/ladder
```
