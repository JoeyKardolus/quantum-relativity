# physics/bench/largest_universe/logic/

What happens: `largest_universe.py` bisects, `probe_size.py` builds, times and drops
one size, `fitting_run.py` builds the engine's universe or refuses it.

## Must not get wrong

- **Refuse before building.** `fitting_run` raises `MemoryError` before anything is
  allocated. On WSL a build that does not fit spills into shared system memory and
  runs very slowly instead of failing.
- **Only `MemoryError` means "does not fit".** Any other error from a build or a step is
  raised; never count a crash as a size that was too big.
- **Drop each universe before the next** (`probe_size` collects and synchronizes), or
  probes stack up in memory and a later one is refused for the wrong reason.
