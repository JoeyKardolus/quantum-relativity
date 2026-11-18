# physics/scenes/gravity/

Phase 1 of the fit (bible `physics/quantum-relativity`, "The fit goes in phases"): only
gravity. One scene, `gravity`, five runs: the uneven universe stands; a lone lump's pull
fades by the law and the lump does not drift; two lumps fall toward each other as
bodies; a kicked lump coasts; a small lump goes round a big one. The viewer opens the
same starts: `python -m physics.view --scene two_lumps | coast | orbit | lone`.

## Must not get wrong

- **Numbers, no verdict.** `gravity` returns what it measured; `definitions/pass_mark.py`
  (written before any search) scores it from outside.
- **A lump is a body only while it keeps its energy.** Every part that follows a lump
  counts what it kept; a lump that fell into one cell or spread away scores 0.
- **A lone lump is run beside the pair**: its drift is taken off the pair's closing.
- **The pull is read beside the law in the same wrapping box** (`newton_tilt`), with the
  twin universe without the lump subtracted.
- **The orbit's kick is worked out from the pull as it stands**, never picked.
- **Phase 1 runs step the layers below axiom 7** (`step(..., quantum=False)`): the
  split store changes nothing in the fluid until a split settles, and at 24 a side it
  takes most of a step's time. `quantum=True` runs every axiom.
- **Placed by hand, on purpose**, like every placed scene; the swirls stay.

## Not done

- A lump is a Gaussian fill of an even grid, not a lump that formed by itself.
- Distances are counted in volumes of the start lattice, not in summed lengths.
