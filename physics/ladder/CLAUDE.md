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
- **Pools never shrink**: capacity stays at its peak (`memory_per_box()["allocated"]`).

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

## Measured (2026-11-16, RTX 5080, wrapped-cell field, quiet_steps 1)

Bytes per box: FULL 8, COARSE 1.75, REUSED 4096 / (512 x holders) (0.125 for 64),
EMPTY 0 plus 0.039 of index. 134M-box grid, half filled: in use 0.89 B a box, first
switch 0.62 s (host hashing of 131k candidates), steady switch 0.048 s, gather with halo
1 0.020 s, exact totals 0.037 s.
Beside the engine it saves nothing yet: the dense boxes stay, so the largest universe at
1 step/s falls from 28^3 cells to 20^3 with it on, while it counts 0.31-0.36 B a box.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/ladder                       # Warp CPU device
LADDER_TEST_DEVICE=cuda:0 PYTHONPATH=. .venv/bin/pytest -q physics/ladder
```
