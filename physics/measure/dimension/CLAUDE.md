# Measure: dimension

How many dimensions the grid has, read only from which energy volumes neighbour which
(axiom 3: a cell has only relations and a size). Two ways of counting:

- **Ball**: count the volumes exactly r steps away. In d dimensions that shell grows as
  r to the power d - 1.
- **Spectral**: let a walker step at random and ask how likely it is to be back home after
  t steps. In d dimensions that falls as t to the power -d / 2.

`definitions/` says what a graph, the reference grid, the ranges and the pass mark are.
`logic/` reads the graph off the engine and measures it. `tests/` proves the whole
measure on plain grids whose dimension is known.

## Must not get wrong

- **No coordinates.** Both dimensions take only who touches whom. Sources come from
  `central_nodes`, never from a cell's place.
- **The reference has the engine's shape.** At the start every volume has a joint to each
  of its 26 neighbours and the grid wraps round. So the grid to compare with is
  `grid_graph(volumes_per_side, 3, wrapped=True, diagonals=True)`, not the 6-neighbour one.
- **One range rule** (`definitions/range_rule.py`) for the reference and the engine alike.
- **The pass mark judges the measure on a reference**, not the engine. A scene returns
  numbers, no verdict.

## Measured (the reference grids)

| grid | ball | spectral |
|---|---|---|
| 2D, 101 a side, 4 neighbours | 2.000 | 2.000 |
| 3D, 41 a side, 6 neighbours | 2.950 | 3.007 |
| 3D, 41 a side, 26 neighbours | 2.981 | 3.745 |
| 3D, 8 a side wrapped, 26 neighbours (a universe 16 cells a side) | 2.972 | 4.003 |

## Not done

- **The spectral dimension fails its pass mark on the 26-neighbour grid** (3.745 and
  4.003 against 3; two tests marked as strict expected failures). The range rule starts the walk at 2 steps
  there, too early for a walker with 26 ways to go. On the open grid of 41 a side,
  walks of 10 to 40 steps give 3.131 and of 20 to 60 give 3.048. The rule was not
  changed to make the test pass: which window is right is a choice to make before a run.
- **A universe 16 cells a side is too small for the spectral dimension at all**: its
  volumes are 4 steps across, and a walk of 9 steps has gone round.
- **After collapses the graph is too small to measure.** The default run has 40 volumes
  after step 1 and 4 after step 9; `measure_dimension` raises (an empty shell, or fewer
  nodes than sources). It says so; it does not return a number.
- **After tears a few volumes neighbour hundreds** (uneven run, step 8: 3,242 volumes, 10
  to 738 neighbours each; ball 3.347, spectral 4.380). Those numbers have no reference
  of the same shape beside them.
