# Axiom 5: packing

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 5):

> Packing: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker with distance; squeezed minus pushes back steeply.

They win over this file and the code.

## Clause by clause

- "more packed is smaller": `size_law` (axiom 3), `packing_law.py`.
- "slower": `clock_rate.py`, `logic/tick_clocks.py`.
- "pulls harder": `pull_source.py`. `logic/pull_fluid.py` is where plus energy feels the pull (`inside_pull.py`).
- "spreads at a finite speed, weaker with distance": `pull_wave_law.py`, `heard_pull.py`, `logic/spread_pull.py`, `logic/start_pull.py`.
- "squeezed minus pushes back steeply": `push_law.py` (quanta to the fourth power), `press_law.py`, `logic/press_joints.py`.

## Assumed

- Packing is quanta over length cubed; one clock law for both kinds. The source is `4 pi G` times packing. The pull is on no cell's clock.
- The pull at a cell's side is its neighbours' mean there, so two volumes agree at the joint they share; `pull_fluid` turns a cell's whole fluid by its own pull and by the slope across it.
- A way's press is tidal; a joint's press is its hardest way.

## Measured (16 a side, seed 1, fix round 3)

- Held run: how a volume's gain goes with its depth is -0.02 at step 1, then up to +0.75; the fuller half of the volumes gains 61.7 million quanta in 24 steps. Gentle run (pull a hundred times weaker): -0.09 to +0.07, and the fuller half still gains 34.1 million. Two tests marked.

## Not done

- **The pull outruns light.** A neighbour is 620 steps of light away. Half of a change is 2 and 4 neighbours away at steps 1,197 and 2,143 (light: 1,240 and 2,480); a hundredth 1 to 4 away at 305, 544, 1,026, 1,268; a trace one neighbour a step (620 times light). Six tests marked `NOT_KEPT`.
- **Changes ring**: a nearer volume is then often the shallower.
- **The default pull turns a box by more than half a turn a step** once energy has gathered (12 rad, uneven run; over 3,000 after the default run's collapse). Two tests marked.
- **A slow clock gathers energy by itself**, pull off: fluid moves slower through a packed cell and piles up. In a universe streaming one way a doubled volume grows from 1,296,000 to 1,861,636 quanta in 40 steps; at full clock rates it coasts away.
- **A joint's clock is never slower** (rate 0.99999 at the collapse limit); `clock` is a ledger no rule reads.
- **The push is a number, not a push** (axiom 2); a joint squeezed and holding shows only in a long run (axiom 6).

## Open

- **The source goes as energy to the fourth power**; what a volume feels goes as its energy.
- **The resting pull deepens with the size of the universe**: -19.3 at 16 a side, -73.0 at 32.
- **The pull's whole turn of a cell is on counter time**, not the cell's clock, so the sinking level all cells share does nothing.
