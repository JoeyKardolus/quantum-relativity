# physics/scenes/run/

How every scene gets its settings and steps its universe. Nothing here measures.

## Must not get wrong

- **A start is settings.** The rebuilt engine takes the start's energies from `Settings`
  (`volume_quanta`, `joint_quanta`, the three unevennesses, `slope`). A fit candidate's
  `start` is a dict of those names and nothing else; any other name raises. There is no
  joint length to pass any more: size comes from energy alone.
- **No rule is switched off here.** `sampled_run` calls the engine's `step` and nothing else.
- **Time is the counter's**: steps taken times `step_time`. A cell's own time is its `clock`.

## Tests

`tests/` runs a start through to a real universe; `definitions/tests/` and `logic/tests/`
check each file alone.
