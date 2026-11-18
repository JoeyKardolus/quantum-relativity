# Axiom 3: structure, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `lay_lattice.py`, `start_grid.py`: the perfect A B A B lattice as tables, built once.
- `relink.py`: after relations changed, the way table follows them. The step calls it with the cells `run_events` (axiom 6) returns.
- `size_cells.py`: the only writer of `length`.
- `check_pattern.py`: counts every way the pattern can break.
- `measure_grid.py`: the grid's size, its cells, `warp` and `lump_share`.
- `read_cells.py`, `read_ways.py`: copies for the viewer.

The last four change nothing. They are here because the engine has only definitions and logic, and a reader is a function, not a thing.

## Must not get wrong

- **The relations are the truth and the ways follow.** `relink` never changes a relation; it rebuilds the ways of the joints the named cells are or touch, keeps the slot of a way that still exists, and hands out `way_pass` so two ways on one mouth never share a pass.
- **`check_pattern` never counts direction codes.** `bare` is a volume with no joint; `fewest_joints` is the fewest different joints round any volume, measured. The old `unwrapped` asked for a joint under each of 26 codes and was filled by listing one joint nine times.
- **The web is joints touching joints.** `web_pieces` joins joints only through joint-to-joint entries; `lone_joints` counts a joint that touches none.
- **`inner_joints` and `crowded_mouths` are measured, not judged**: a joint all of whose volumes have merged into one, and a mouth more than one way goes through.
- **`warp` is the reach of a lump's ways over the reach of all ways**, joints included; 1 in an even grid. A measure, not a pass mark.
- Logic imports definitions of any axiom and never another axiom's logic.
