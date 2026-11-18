# physics/scenes/volumes/definitions/

Pure numbers: no universe is read here.

- `power_spectrum` treats its cube as wrapping, as the universe does. It is trusted
  because it recovers the slope of the engine's own `uneven_field` (the test).
- `largest_group` counts members only; a pair with a non-member end joins nothing.
- `LUMP_FACTOR` is a constant written before any run. Changing it changes what every
  scene calls a lump: do not.
