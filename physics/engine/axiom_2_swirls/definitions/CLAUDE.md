# Axiom 2: swirls, definitions

What things are under this axiom. Each file's docstring is its contract.

- `swirl_arrays.py`: `swirl_axis`, `swirl_sense` per cell. They remember how the cell was made; the swirl itself is in the fluid's phase.
- `swirl_phase.py`: the phase of each box of a cell that is one whole swirl.
- `sign_law.py`: `makes_pull`, `feels_pull`, `pushes`.

## Must not get wrong

- **The phase falls the way the swirl goes.** Fluid flows from the wave that is ahead (`pair_rotation`, axiom 1), so `swirl_phase` is minus the sense times the angle round the axis.
- **One whole turn.** Going once round any ring the phase changes by exactly one turn. Half a turn would leave a jump where the ring closes.
- **Level toward the sides.** A box on the outside of the block has the phase of the box behind it. Without that the swirl throws fluid at the block's flat sides and every volume leaks through four of its six faces.
- **The sense is no label.** A cell whose phase is wound the other way goes round the other way; a test pins it.
- **No kernel walks a ring.** The old step moved quanta from box to box along rings and left the phases behind; that stirred the fluid and pumped nine times the crossing. It is gone; do not bring it back.
- **One sign law.** Every rule that makes, presses with or feels the pull, or pushes, asks `sign_law.py`. No charge, no cell that does both.
- Definitions import only definitions, of axiom 3 and this one.
