# physics/measure/

Measures read from the foam's relations alone (bible `physics/quantum-relativity`, rule 1 and
"How results are trusted"). Host numpy and scipy; no engine import, no GPU. Module map in
`__init__.py`. Scenes call these; a measure never runs the engine itself.

## Must not get wrong

- **Validated first.** A measure is used on the foam only after its test recovers a known
  answer: ball and spectral dimension within 10% of 2 and 3 on plain grids (`grid_graph`); the
  whirl measures on a placed whirl of known turns and direction. A new measure lands with its
  reference test, and a scene reports the reference's own number beside the foam's (a small
  graph pulls both estimates off the true value).
- **No coordinates in the graph measures.** Ball and spectral dimension take only an
  adjacency matrix; sources come from `central_nodes` (far points), never from box places.
- **Ranges inside the graph.** Shells and walks must not reach the graph's edge: the rule used
  in the tests and `scenes/space_dimension.py` is R = a quarter of the source's farthest
  distance, radii 2..R, walk times R^2/10..R^2. Change it in both or neither.
- **Winding is whole by construction** (wrapped phase steps round a closed loop), so "is it an
  integer" is never the test; whether it stays equal to the placed turns, and how far the flow
  circulation and the strength sit from it, are.
- **Signs.** Squares and loops are walked right-handed about +normal (b = normal + 1,
  c = normal + 2); a whirl's winding vector points along its line.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/measure
```
