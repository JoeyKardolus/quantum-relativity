# physics/scenes/dimension/logic/

- `graph_dimension` hands the engine's `read_ways` and kinds to `physics.measure`'s
  `volume_graph` and `measure_dimension`; it adds nothing of its own.
- `space_dimension` measures before the first step and after the last. A universe that
  has fallen into too few volumes makes the measure raise; the scene does not catch it:
  a grid with no room has no dimension to report.
