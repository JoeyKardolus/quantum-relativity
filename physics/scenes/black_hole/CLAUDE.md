# physics/scenes/black_hole/

Scorecard scene 3.

**A horizon is not the core: it is the affected region round it** (owner, 2026-11-17).
The core's own size shrinks as its energy grows (size law); measuring that gave a power
of -1 and was the wrong measure.

**Known answer**: the region's radius grows in proportion to the core's energy (power +1).

**Pass mark** (written 2026-11-17 before the run): the least-squares power of held
radius against the core's energy lies in 0.75 to 1.25, over at least 4 energies
spanning a factor 8 or more whose radius is above 0 and below the box's limit. The
depth radius is a second reading, reported beside it under the same band.

## Must not get wrong

- **Held region**: volumes sorted in shells by how many ways lie between them and the
  core; boundary j closes the core and shells 1 to j. The region reaches as far as every
  boundary from 1 on took net plus energy in over the window. The core's own skin,
  boundary 0, is reported apart (`core_gain`).
- **Every reading has the twin subtracted**: the same start with no core. The even
  grid's own jitter moves energy and tilts the pull.
- **A slow pull has not arrived**; a packed core's clock is slow. Read `core_clock`
  before believing a radius.

## Assumed (for the owner)

- **Shells are cubes.** Every joint of the pattern is a real joint now, so a volume has
  ways to 26 others and "ways between" is the largest of the three offsets. The old
  scene counted 6 ways.
- **The depth level is the twin's own pull scatter.** The old level, the even grid's
  heat per quantum, is gone: heat is now a count of quanta, not an energy. The depth
  radius is where the core's depth falls to the size of the even grid's own unevenness.
