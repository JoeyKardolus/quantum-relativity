# The fitted settings

What the fit chose, kept apart from the engine's defaults.

- `definitions/fitted_values.py`: the values and what the search found.
- `logic/load_fitted.py`: the values as engine `Settings`.

## Must not get wrong

- **Nothing is fitted** (2026-11-18): no search has run on the rebuilt engine, so the
  values are empty and the fitted universe is the default one.
- **Only a point that holds is ever written here**, with the search that chose it
  (sizes, seeds, short and long scores) in `fitted_on`.
- **Never change a default.** A scene that wants the fitted universe asks for it by
  name: `load_fitted()["settings"]`.
- **A fitted value is a candidate** until the unfitted scenes pass with it.
- **Every fitted name must be a searched setting inside its range**; a test checks it.

## Deleted with the rebuild

- `fitted/settings.json` and its two candidates from the 2026-11-17 search: they named
  settings the engine no longer has (crossing_rate, seal_thin, fluid_pressure,
  joint_length).
- The separate "start" part: the start's energies are engine settings now.
