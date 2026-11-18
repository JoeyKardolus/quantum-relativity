# physics/scenes/two_lumps/definitions/

- `newton_tilt` is differenced over the same two places beside the lump as the engine's
  pull is, so the two are one measurement of two fields. In a huge box it is
  1 / distance squared; in a real one the copies weaken it, more at larger distances,
  to 0 at half the box. Until 2026-11-18 the copies were summed bare and it read 2e-4
  there in a box of 100, a third of the free value: every earlier `newton` number and
  `newton_power` was read beside that.
- `extra_centre` weighs places by extra energy only; with none it returns 0, not NaN.
- The mark has two parts with their own functions so a result can say which one missed.
