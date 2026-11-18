# physics/view/source/definitions/

What a source is.

- `source_interface.py`: `Source`, a Protocol: `reset()`, `step()`, `steps`,
  `view(layer=None)`, `stats()`. The stepper and the saved pictures ask for nothing else.
- `stat_names.py`: `STAT_NAMES`, the nine counts the title reads.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine. A test double on plain arrays
  must fit `Source` (`physics/view/tests/grid_cases.py` has one).
- A count missing from `stats()` shows as "?" in the title. Never fill one in.
