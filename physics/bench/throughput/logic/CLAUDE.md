# physics/bench/throughput/logic/

What happens: `throughput.py` warms up, then times steps; `engine_run.py` builds the
engine's default universe as a `Run`.

## Must not get wrong

- **Synchronize at both ends.** GPU work is queued; without `wp.synchronize()` before
  the clock starts and before it stops, the time is the time to queue, not to run.
- `throughput` knows nothing of the engine. Only `engine_run.py` imports it.
- `engine_run` uses the engine's default `Settings`, every rule on. Do not switch a
  rule off to get a faster number.
- `bytes_per_cell` reads Warp's whole pool. With a second universe alive in the same
  process it counts that one too.
