# Measure: dimension, logic

What happens when the dimension is measured. Each file's docstring is its contract.

- `volume_graph.py`: turns the engine's ways (`read_ways`) and kinds (`read_cells`) into
  the graph of energy volumes. Two volumes touch when a way goes between them.
- `central_nodes.py`: the nodes deepest inside a graph, found from far points.
- `ball_dimension.py`: 1 + the slope of shell size against radius.
- `spectral_dimension.py`: -2 times the slope of a lazy walk's return chance against time.
- `measure_dimension.py`: picks the sources, takes the ranges of the range rule, runs both.

## Must not get wrong

- **`volume_graph` is the only file that knows the engine.** It reads host rows; it holds
  no state and steps nothing. The others take any adjacency matrix.
- **A way inside one volume joins nothing.** After a collapse a joint can have the same
  volume at both ends; that way is left out. A way ending on anything but a living energy
  volume raises: the engine's pattern is broken, and the measure does not hide it.
- **Node n is volume `volumes[n]`.** Ids have gaps (joints, dead cells); always go through
  the returned list.
- **The walk is exact**, not sampled: the chance is spread step by step. It is lazy (stays
  put half the time) so that it does not tick between odd and even steps on a grid.
- **Shells that run off the graph raise.** `ball_dimension` refuses an empty shell rather
  than fit through it.
- Logic imports definitions; no file here imports the engine's logic.
