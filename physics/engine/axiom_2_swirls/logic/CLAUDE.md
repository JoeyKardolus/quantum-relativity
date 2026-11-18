# Axiom 2: swirls, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `start_swirls.py`: at the start every living cell gets an axis and a sense drawn from the seed, and its fluid's phase is wound once round that axis.

That is all. In a step nothing here runs, so nothing keeps a cell a swirl: see "Not done" in the axiom's overlay.

## Must not get wrong

- **No rule turns a swirl.** If a cell stops going round, the fluid did that (the crossing, the pull); do not add a step that sets it going again without the owner's word on what a swirl's whole steps are.
- **Start swirls writes phases, so it runs after `start_energy`** and before anything reads a phase.
- **Every cell, joint and volume alike**, and each from its own draw: the grid has no chosen direction.
- **A tear hands the axis and sense on** (axiom 6) and copies the phases layer by layer: a piece is wound only if the volume still was.
- Logic imports definitions of any axiom and never another axiom's logic.

## Tests

`tests/` runs the fluid alone on a real 4-a-side universe: 40 steps (no longer), every cell goes round its own way and no quantum leaves its cell.
