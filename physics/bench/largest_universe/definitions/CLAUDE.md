# physics/bench/largest_universe/definitions/

What things are: the pass mark (`pass_mark.py`), the memory a size needs
(`memory_law.py`), the sizes to try (`size_ladder.py`), the records (`search.py`), what
stopped a search (`limits.py`).

## Must not get wrong

- Nothing here runs anything or imports `logic/`.
- **The pass mark is one step a second**, written before any run. Never lower it to get
  a bigger universe.
- **`BYTES_PER_CELL_ID` is measured, not chosen.** After a change to the engine's state,
  measure it again (`facts()["bytes_per_cell"]` over 1 + `SPARE`) and keep the largest
  value seen, so the guard refuses too early and not too late.
- A side must be even and at least 4, or the engine's A B A B pattern cannot wrap.
- Speed is named before memory when both stopped a search.
