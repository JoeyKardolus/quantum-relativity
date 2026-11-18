# physics/bench/throughput/definitions/

What things are: the universe the bench times (`engine_start.py`), a universe ready to
be timed (`run.py`), the result and its formula (`rates.py`), the timing plan
(`timing.py`).

## Must not get wrong

- Nothing here runs anything, imports `logic/` or imports the engine.
- `SEED`, `SPARE`, `STEPS` and `WARMUP` are fixed before a run. Changing one changes
  every number measured before; write the new numbers in `../../CLAUDE.md`.
- `SPARE` is also read by the memory law of `largest_universe`. It must be the share
  handed to the engine's `new_universe`.
- `Run.cells` is the count at the start. `Run.facts` is read after a timing, never
  inside it.
