# physics/scenes/volumes/logic/

Readers: each takes a universe and returns host numbers; none writes to it or steps it.

- Ids at or above `side ** 3` were made by tears. `volume_field` cannot place them;
  `observer_cells` and `count_lumps` do count them, since they are volumes like any other.
- `count_lumps` measures against `even_volume_quanta(settings)`, the start's even
  volume, not against the mean now: the total is not fixed.
- A dead way slot (`way_joint` -1) is already left out by the engine's `read_ways`.
