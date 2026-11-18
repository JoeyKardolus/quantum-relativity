# physics/scenes/dimension/

Scorecard scene 8. The measures themselves live in `physics/measure/dimension/`.

**Known answer**: 3.

**Pass mark** (written 2026-11-17 before any run): ball dimension and spectral
dimension at the end both within 10% of 3 (2.7 to 3.3). The plain wrapping grid's own
numbers at this size are reported beside.

## Must not get wrong

- **Only relations are read**: which volume has a way to which (axiom 3: no geometry).
  No length, no energy, no coordinate.
- **Never read a dimension bare.** A small graph pulls both estimates off 3; the
  reference grid of the same size and shape says by how much.
- **The reference is the 26-neighbour grid.** In the rebuilt engine every joint of the
  pattern is a real joint, so a start volume has ways to 26 others, not 6. A 6-neighbour
  reference would no longer be the start's own graph.
- While no tear or collapse has happened the end equals the start equals the reference:
  that is the plain grid's number, not a finding about space.

## Not done

- At 8 places a side every volume is within 2 steps of every other; the measure has no
  room. Run at 16 or more.
