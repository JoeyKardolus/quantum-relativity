# The fitted settings

What the fit chose, kept apart from the engine's defaults.

- `definitions/fitted_values.py`: the values and what the search found.
- `logic/load_fitted.py`: the values as engine `Settings`.

## Must not get wrong

- **Phase 1 (gravity) does not hold** (2026-11-18): the values are the best candidate
  (score 0.10 of 1), kept so the viewer's gravity scenes can be watched. Its measured
  parts are in `fitted_on`.
- **A value here is a candidate until its phase holds**; `status` says which. It is written with the search that chose it
  (sizes, seeds, short and long scores) in `fitted_on`.
- **Never change a default.** A scene that wants the fitted universe asks for it by
  name: `load_fitted()["settings"]`.
- **A fitted value is a candidate** until the unfitted scenes pass with it.
- **Every fitted name must be a phase 1 setting inside its range** (`gravity_ranges.py`); a test checks it.

## Deleted with the rebuild

- `fitted/settings.json` and its two candidates from the 2026-11-17 search: they named
  settings the engine no longer has (crossing_rate, seal_thin, fluid_pressure,
  joint_length).
- The separate "start" part: the start's energies are engine settings now.
