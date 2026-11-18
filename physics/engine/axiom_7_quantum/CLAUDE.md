# Axiom 7: quantum

The owner's words are axiom 7 in `docs/bible.md#physicsquantum-relativity`, "The axioms". Read them there, word by word.

They win over this file and the code.

## Clause by clause

- "waves add and cancel": `phase` in `pair_rotation` (axiom 1).
- "split over several routes": `open_splits`. New outflow of a volume that went into two or more volumes in one step met a fork; nothing is staged.
- Oscillation and pointer: `turn_splits`, `speed_draw`, `pointer_law`.
- "something reacts": a collapse or tear in a half's cell (`carry_events`), or a scene's detector (`react`).
- "loses its momentum": `carry_on`. A half goes on with its cell's fluid; at every crossing friction takes direction off the split (`directed`). At 0 it settles.
- Settling whole: `settle`, `draw_whole`.

## Assumed (for the owner to confirm)

- One moving energy is what one volume sends out through a fork; while its split is open, the volume's later new outflow is more of it.
- Friction on other energy is no reaction; friction on the split's own energy is its loss of momentum.
- Settling is coming to rest: what is drawn joins the pointer's cell at the phases that cell has.
- The direct pull is the draw at settling, at once, at any distance.

## Measured (seed 1, fix round 3)

- Gentle settings, 8 a side, 1,500 steps: all 64 volumes open a split in step 1 and by step 100 the splits claim 97% of the plus energy, none of it twice (`over_claimed` 0; at friction 1e-9 too, where the old store claimed 269%). `directed` falls by friction alone: 7.5 million (step 50), 365,319 (200), 26,924 (400), 2,653 (1,200). The first splits settle at steps 1,296 and 1,338 and draw 640,597 and 771,745 quanta, about a mean volume each, into one cell. Nothing short; books exact.
- Uneven run: tears react, and hundreds of splits settle a step.

## Not done

- **No pull acts between the halves before settling.**
- **A collapse after the first steps takes quanta halves still claim**: collapsing run, step 9: 4 splits not whole, a settling 9,367 quanta short (two expected failures).
- **Phase 1 of the fit steps without this axiom** (`step(..., quantum=False)`): until a split settles the store changes nothing in the fluid, and it takes most of a step's time.
- **The oscillation changes no outcome**: at 1 to 64 turns a step the pointer jumps every step, each jump a fresh draw by share. It turns per engine step, which is a time.
- **One split per volume, not per fork**: two forks of one volume are one split. A split per fork is about 200,000 open splits at 16 a side; that needs the store on the GPU.

## Open

- **Somebody can know the speed**: it follows from the seed, the split's step and its volume.
- **Faster than light, and lumpy**: a settling moves a volume's worth of energy across the whole universe in one step.
