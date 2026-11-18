# physics/scenes/run/definitions/

What a scene's start is: a set of start amounts in the engine's `Settings`.

- `START_AMOUNTS` lists the only names a `start` may hold. Add a name only when the
  engine's `EnergySettings` gains a start amount.
- `even_volume_quanta` is `volume_quanta` times the 216 boxes of a block. The start's
  true mean is about half a percent above it (the box-to-box unevenness is not centred);
  a scene that needs the exact mean reads the state.
- Nothing here imports `logic/` or touches a universe.
