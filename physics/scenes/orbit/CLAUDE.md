# physics/scenes/orbit/

Scorecard scene 1 (Kepler).

**Known answer**: period squared over distance cubed is the same at every distance.

**Pass mark** (written 2026-11-17 before any run): 32 places, a big volume of 8 times
and a small one of 2 times, 2, 3 and 4 volumes apart, 2,000 steps. The small one's extra
energy goes at least one whole turn round the big one at two or more distances before
half of it is lost, and period squared over radius cubed agrees between those distances
within 10%. No whole turn anywhere is a miss.

## Must not get wrong

- **A moving mass is a pattern of energy travelling through the grid**, never a volume
  travelling. The blob is read as extra quanta on the volumes' lattice.
- The big volume and the 26 volumes round it are left out of the blob: what the big one
  sheds to its neighbours is not the small one.
- The blob is read above a twin run (the big volume alone, same seed, same step), not
  above the start: the even grid moves energy by itself.

## Assumed (for the owner)

- **Sideways motion is a phase ramp.** In the rebuilt engine fluid flows toward where
  the phase runs on, so the small volume's fluid is started with a ramp of `SIDEWAYS`
  (a twelfth of a turn a box) across the line to the big one. The old scene started at
  rest because the old crossing took all direction off. The number was picked before
  any run; nothing says it is an orbit's speed.

## Not done

- The engine's own finding: one volume given a ramp in a still universe draws energy in
  and does not coast (the ramp does not run on into the next cell). Until that is
  settled the scene cannot be expected to go round.
