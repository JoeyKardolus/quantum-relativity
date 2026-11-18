# Axiom 2, logic

`read_swirl.py` reads a cell's boxes on the host, one function a measure: `whole_turns`, `round_flow`, `core_share` (each about one direction), `off_rule`.

- `whole_turns` is counted on the inner rings of the outer layers; `None` when they disagree. That is a finding, not an error to hide.
- The phase falls the way the fluid goes, so the whirl reads -1 turn about each direction; turned half round, +1 about the other two.
- `off_rule` is 1 for plain counting noise. That mark is fixed.
- It is for titles and tests. Never call it inside a step.
