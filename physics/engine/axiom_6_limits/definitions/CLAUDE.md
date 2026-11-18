# Axiom 6: limits, definitions

What things are under this axiom. Each file's docstring is its contract.

- `limits_arrays.py`, `limits_settings.py`: `fresh_quanta`, `made`, the flags; `squeeze_rate`, `collapse_thin`, `tear_length`.
- `make_law.py`: what a squeezed joint makes in its own time.
- `collapse_law.py`: a joint holding `collapse_thin` times its fresh quanta collapses.
- `tear_law.py`, `tear_map.py`: a too-long volume tears when each of its three pieces would hold something; how the block is dealt.
- `held.py`: the host's copy of the cells one step's events touch.

## Must not get wrong

- **`make_law` goes toward the level at `squeeze_rate` per unit of the joint's own time**: `level * (1 - exp(-rate * tau))`. It never passes the level and the rate always counts. In a real run a joint's clock rate is 1 to four places, so the local clock changes nothing one can measure there.
- **`collapse_law` does not ask for a squeeze now.** A joint's quanta only rise while squeezed, so being past the limit is having been squeezed too far.
- **`tear_law` needs all three pieces to hold something**, counted across the volume's swirl axis. A tear with an empty middle makes no joint.
- **`fresh_quanta` is never changed** after a joint is made.
- **`tear_map` keeps every quantum** and flips only the middle two layers. Phases are float64, in `held` too.
- Definitions import only definitions.
