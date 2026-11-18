# Axiom 4: crossing

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 4):

> Energy diffuses between plus swirls only through minus; that is the only friction, and it becomes heat.

They win over this file and the code.

## Clause by clause

- "Energy diffuses between plus swirls": `logic/cross.py`. Two boxes that face each other across a joint take the fluid's own step (`pair_rotation`, axiom 1). A fuller box evens out with an emptier one; fluid flows from the phase that is ahead.
- "only through minus": only along a way whose joint is a living joint that holds minus energy, on the slowest of the three clocks (`crossing_clock.py`).
- "that is the only friction": `friction_law.py`, called only in `cross`. A share (`friction`) of what crosses loses its direction in the joint and arrives at rest in the fluid it joins (`arrival_law.py`).
- "and it becomes heat": `heat_quanta`, a whole count per cell, raised only by friction.

## Assumed

- `friction` is one share for every joint.
- The phases of two cells are read as they stand: one fluid, one phase field.

## Measured (fix round 3)

- A universe of one phase and even energy: not one quantum crosses, at any friction. One volume doubled in that still fluid (1,296,000 among 648,000) has lost 164,475 quanta by step 10 and 684,755 by step 50, and stays level.
- The fluid a flow joins turns less the more friction there is.
- Held and gentle runs: half of what crosses loses its direction; heat is 51% of the plus energy at step 24.

## Not done

- **On a real start the volumes get lumpier**, pull as good as off: spread of their quanta 66,124 at step 1, 316,367 at step 33 (gentle run).
- **The joint's own fluid plays no part**, and energy crosses it in one step whatever its length (see axiom 1).
- **Friction does not bring the fluid to rest; heat is a count.** A box has one phase for all its quanta. Pull off, 400 steps: the flow between boxes inside the volumes rises from 22.6 to 43.7 million at friction 0.5, to 48.9 million at friction 1: the phases grow disordered.
- **A cell does not coast by its own tilt.** Which way energy crosses is set by the phase step across the joint. One volume given a phase ramp toward +x in a still universe lags its neighbours and draws energy in: +115,473 quanta in step 1, +856,323 by step 40. Where the phase runs on from cell to cell a lump does coast: in a universe streaming toward +x with every clock at full rate, a doubled volume hands 175,618 quanta to its +x neighbour in 10 steps, 29,503 are two volumes on by step 20, and the -x neighbour loses. For the owner: the old side law coasted from a tilt alone but made motion out of nothing.
- **A settling (axiom 7) moves energy through no joint** (`off_ways`).
