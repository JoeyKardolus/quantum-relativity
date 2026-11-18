# Axiom 2: swirls

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 2):

> Cells are swirls: plus only pulls, minus only pushes, both oscillate in whole steps.

They win over this file and the code. `tests/check_step.py` checks the axiom on real runs.

## Clause by clause

- "Cells are swirls": `logic/start_swirls.py` makes every cell, volume and joint, a standing swirl (`swirl_phase.py`, `swirl_share.py`): the slowest standing wave of its six boxes along two directions, a quarter turn apart, so its phase goes once round the axis and its core stands open. No rule turns it afterwards: `flow_inside` (axiom 1) carries it.
- "in whole steps": the phase comes back to itself after one whole turn round the axis.
- "plus only pulls": `pull_source` asks `makes_pull`, `pull_fluid` asks `feels_pull` (axiom 5).
- "minus only pushes": `press_joints` asks `pushes` (axiom 5).

## Assumed

- One axis for the whole grid, drawn from the seed, and neighbours of opposite sense (2026-11-18). Two volumes then hold equal waves in the boxes that face each other across a joint, so the swirls mesh like gears and nothing crosses a joint by itself. Own axes (before) ground on each other, unwound within 20 steps and carried no motion from cell to cell.
- "Oscillate" is read as going round.
- A torn volume's pieces keep its axis and sense; a merged volume keeps the kept one's.

## Checked at every step (`tests/check_step.py`)

Read off the fluid, not off a label: `against_*` (a cell whose own flow round its axis does not go its sense's way), `unwound_*` (the phase no longer goes exactly once round).

## Measured (16 a side, seed 1)

- Default, held and gentle runs: every volume keeps its swirl at every step (none against its sense, none unwound). Even start, 200 steps: about 1e-4 of the plus energy crosses a step, heat 0.001 (own axes: 9%, heat 0.1).
- Joints keep theirs in the default run. At push 40 the balance leaves a joint one quantum a box and the rounding of whole quanta unwinds it (1,810 of 3,584 against their sense).

## Not done

- **No rule makes or keeps a cell a swirl.** It stands only while nothing disturbs it: lost through collapses and tears (a tear copies whatever phases are left) and in joints of few quanta. Nine run tests are expected failures.
- **A lump is not a standing swirl**: placed extra energy and the pull's own slope across a cell unwind it there.
- **The grid has a chosen direction**: one swirl axis for all cells.
- **Nothing oscillates**; `check_step` reads the middle ring only.

## Open

- **The push moves nothing.** A joint's push is what its press is divided by; the joint holds or yields. With no geometry there is nothing to shove but fluid, and the axioms name no way it shoves fluid.
- **`pushes(kind)` decides nothing**: only a joint lies in a way, so only a joint can bear a press.
