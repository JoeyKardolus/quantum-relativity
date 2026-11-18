# physics/scenes/two_lumps/

Scorecard scene 12, gravity: "does gravity work".

**Known answer**: the pull falls as one over distance squared; two masses fall toward
each other.

**Pass mark** (written 2026-11-17 before any run): 48 places, two volumes packed 4
times, 2, 3, 4, 6 and 8 volumes apart, 2,000 steps.
Fall-off: the pull's power from 3 volumes on is within 0.2 of the 1 / d squared law's
power measured the same way in the same box.
Falling: at every distance the two lumps' extra energy closes in by more than the
scatter of a lump alone, and more than half of it is still in the two lump volumes at
the end (a lump that has spread away has not fallen). Gravity works only if both pass.

## Must not get wrong

- **The pull is read beside its law in the same wrapping box**, as a ratio to the first
  distance (`newton`): the box's copies pull too.
- **The run with the second lump not there is subtracted** from the tilt: the even
  grid's own jitter tilts the pull.
- **Distance is the summed lengths of the cells between the lumps' middles** at the start.

## Changed with the rebuilt engine

- The old fall-off mark was judged on a run with crossing off, so the lumps kept their
  energy. The engine no longer has an off switch (axiom 4 has no rate setting), so the
  fall-off is judged with every rule on. That is a harder mark; it was not rewritten to
  be met.
