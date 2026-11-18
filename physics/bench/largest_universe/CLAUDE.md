# physics/bench/largest_universe/

The biggest universe that fits the GPU and still runs at one step a second. File map
in `__init__.py`.

- `definitions/` holds the pass mark, the memory law, the sizes to try, the records and
  the three limits.
- `logic/` bisects the sizes, probes one size, and builds the engine only when it fits.
- `tests/` runs the whole search on the real engine. They skip without CUDA.

## Must not get wrong

- **The search bisects**, so it takes a bigger universe to be never faster. If that
  stops being true the answer is wrong; time each size alone with `--sizes`.
- **Every result names its limit**: speed, memory, or the size list ran out. "Memory"
  may only mean another program held the card. It is not a result about the engine.
- A probe times the first few steps from the start. A universe that tears a lot grows
  slower later; the search does not see that.
