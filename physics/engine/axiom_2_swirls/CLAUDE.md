# Axiom 2: swirls

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 2):

> Cells are swirls: plus only pulls, minus only pushes, both oscillate in whole steps.

They win over this file and the code. `tests/check_step.py` checks the axiom on real runs.

## Clause by clause

- "Cells are swirls": `logic/start_swirls.py` gives every cell, volume and joint, an axis and a sense from the seed and winds its fluid's phase once round that axis (`definitions/swirl_phase.py`). Fluid flows from the phase that is ahead, so the cell's own fluid goes round. No rule turns it afterwards: `flow_inside` (axiom 1) carries it on the cell's local time.
- "in whole steps": the phase comes back to itself after one whole turn round the axis.
- "plus only pulls": `pull_source` asks `makes_pull`, `pull_fluid` asks `feels_pull` (axiom 5).
- "minus only pushes": `press_joints` asks `pushes` (axiom 5).

## Assumed

- One whole turn, the same in all six layers, at the start.
- "Oscillate" is read as going round.
- A torn volume's pieces keep its axis and sense; a merged volume keeps the kept one's.

## Checked at every step (`tests/check_step.py`)

Read off the fluid, not off a label: `against_*` (a cell whose own flow round its axis does not go its sense's way), `unwound_*` (the phase no longer goes exactly once round).

## Measured (16 a side, seed 1)

- Fluid alone: every cell goes round its own way for 40 steps (the test's length). A joint loses its swirl by itself later: 268 of 448 unwound at step 2,000.
- Gentle run (pull as good as off): volumes unwound at steps 9, 13, 21: 79, 223, 354 of 512; 121 go against their sense at step 21. Held run: 424, 485, 498; 284 against. Joints keep theirs in both; in the uneven run 179 joints are unwound by step 5. Ten run tests are marked `NOT_KEPT`.

## Not done

- **No rule makes or keeps a cell a swirl.** It is wound at the start and the fluid undoes it at any pull strength: facing boxes of two volumes trade by their phases (axiom 4). A tear copies whatever phases are left.
- **A joint's swirl does nothing.** Nothing outside a joint reads its fluid.
- **The swirl drives energy through the sides**: with the volumes' phases flat nothing crosses in an even universe; with swirls 422,860 quanta cross in step 1 (8 a side).
- **Nothing oscillates, and nothing is counted in whole steps.** Going round one way is no swing, and any phase pattern closes after a whole number of turns. `check_step` reads the middle ring only.

## Open

- **The push moves nothing.** A joint's push is what its press is divided by; the joint holds or yields. With no geometry there is nothing to shove but fluid, and the axioms name no way it shoves fluid.
- **`pushes(kind)` decides nothing**: only a joint lies in a way, so only a joint can bear a press.
