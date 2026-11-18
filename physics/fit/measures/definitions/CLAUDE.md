# Measures, definitions

What the fit's words mean, as constants and formulas. Each file's docstring is its contract.

- `lump_marks.py`: the numbers that say what a lump is (`LUMP_FACTOR`, `CENTRES`, `SPREAD_LUMPS`).
- `lump_volumes.py`: which energy volumes are lump volumes.
- `lump_groups.py`: lump volumes joined through ways are one lump.
- `overstretched.py`: which cells are longer than the tear length.
- `region_contrast.py`: how uneven the energy is over regions bigger than a cell.
- `observer_time.py`: the time that passes on the clock where the energy sits.

## Must not get wrong

- **These are marks, not settings.** They were written down before any search and are never fitted. Change one only before a search, and say what the old one let through.
- **A lump is counted against the start.** A lump volume holds at least twice the start's mean volume energy, so "lump" means the same thing at every step of a run.
- **Pure functions on arrays.** Nothing here touches the engine's state or the GPU; only the engine's kind constants are imported.
- Definitions never import logic.
