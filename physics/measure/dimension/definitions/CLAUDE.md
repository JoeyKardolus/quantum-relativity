# Measure: dimension, definitions

What things are. Each file's docstring is its contract.

- `adjacency.py`: a graph is a list of touching pairs, kept as one symmetric 0/1 matrix.
  A pair listed twice counts once.
- `grid_graph.py`: the reference, a plain grid of known dimension. `diagonals=True` gives
  each node its 8 (2D) or 26 (3D) neighbours; `wrapped=True` closes it round on itself.
- `range_rule.py`: how far shells and walks may go: a quarter of the way to the farthest
  node, never under 3 steps.
- `pass_mark.py`: `WITHIN = 0.1` and `passes`: a measure must come within a tenth of a
  known dimension.

## Must not get wrong

- **Unweighted.** A graph holds only who touches whom: no length, no energy, no count of
  ways between two volumes.
- **Nodes of `grid_graph` are numbered in raster order**, the last coordinate fastest. The
  engine's start numbers its volumes the same way, which is why the two can be compared
  entry by entry.
- **The numbers in `range_rule.py` and `pass_mark.py` were written before any run.** Never
  change one to make a measure pass; see "Not done" in `../CLAUDE.md`.
- Definitions import only definitions, and nothing from the engine.
