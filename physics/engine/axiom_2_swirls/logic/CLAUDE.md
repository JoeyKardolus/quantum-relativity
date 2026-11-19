# Axiom 2, logic

`read_swirl.py` reads a cell's boxes on the host: `whole_turns`, `round_flow`, `flow_along` (each for one ring), `still_share`, `off_rule`. `turn.py`: `turn`, `turn_of`. `flip.py`: a scene's act.

- `whole_turns` is counted on every row round a ring's core; `None` when they disagree. That is a finding, not an error to hide.
- The phase falls the way the fluid goes: a volume reads -1 in each ring, a joint +1.
- `off_rule` is 1 for plain counting noise. That mark is fixed.
- `read_swirl` is for titles and tests. Never call it inside a step.
- `turn` runs right after the fluid's own step.
