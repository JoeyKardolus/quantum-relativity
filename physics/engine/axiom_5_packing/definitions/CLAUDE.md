# Axiom 5: packing, definitions

What things are under this axiom. Each file's docstring is its contract.

- `packing_arrays.py`, `packing_settings.py`: packing, rate, clock, the pull and its ledgers, press and squeeze; light speed, the two strengths, the clock's packing.
- `packing_law.py`, `clock_rate.py`: quanta over length cubed; more packed is slower.
- `pull_source.py`: only plus energy is a source, by its packing.
- `pull_wave_law.py`, `heard_pull.py`: the wave step, the pull at rest, what a cell hears of its neighbours.
- `inside_pull.py`: where a box stands between its cell's middle and its sides.
- `press_law.py`: what a joint bears of its two volumes being pulled together.
- `push_law.py`: the push and the squeeze.

## Must not get wrong

- **The pull is float64.** In float32 one wave step of a small change was under what the number could hold and was thrown away: a volume could change by a thousandth and the pull never knew.
- **No clock rate in the wave.** The pull spreads between cells; a packed cell's slow clock must not make it pull less.
- **The pull slopes across the whole cell.** `inside_pull` gives every box its place between the middle and the sides; the eight weights sum to 1. A step at the skin only (the old way) is a wall to climb, not a pull.
- **The press is tidal.** Two volumes falling together down one even slope press nothing: the law is the reduced quanta times the sum of the two slopes toward the joint.
- **An empty cell is not heard**, hears nothing and is not read: `heard_pull`, the press and `pull_fluid` all leave it out.
- **One pull.** Every rule reads `pull` and only differences of it.
- **The push goes as the joint's quanta to the fourth power.**
- The source goes as energy to the fourth power while what a volume feels goes as its energy: open, see the axiom's overlay.
