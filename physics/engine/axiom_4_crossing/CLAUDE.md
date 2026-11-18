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

## Measured (2026-11-18)

- A universe of one phase and even energy: not one quantum crosses, at any friction.
- The fluid a flow joins turns less the more friction there is.
- Held and gentle runs: at least 0.65% of the plus energy crosses a step (the meshing swirls send little by themselves).
- A lump with a phase step of 0.3 a box across its whole fluid, even fluid of one phase: 6.6 boxes in 50 steps (the free fluid's own speed: 7.4). With meshing swirls: 4.2 boxes; with each cell wound round its own axis (before): 2.2, then back.

## Not done

- **A lump coasts only while it holds together.** Motion crosses a joint with the energy (the pair's own step; friction takes its share), but nothing but the pull holds a lump, and a pull strong enough for that packs it into single cells, which stay where they are (`scenes/gravity`).
- **The joint's own fluid plays no part**, and energy crosses it in one step whatever its length (see axiom 1).
- **Friction does not bring the fluid to rest; heat is a count.** A box has one phase for all its quanta.
- **Only 16 of a face's 36 boxes trade straight across**; the 20 at its rim trade with the neighbours across edges and corners (`mouth.py`).
- **A settling (axiom 7) moves energy through no joint** (`off_ways`).
