# physics/scenes/spin/definitions/

- `spin_cells` picks cells along the start lattice's diagonal, 4 places apart, so no
  two watched cells are neighbours; each joint is the face joint just after its volume
  along the third axis. It is a fixed rule, not a search for cells that hold.
- `turn_record` counts a step as held only when every layer reads the expected turns.
  Five layers right and one wrong is not held.
- `NOT_ASKED` lists what the known answer includes and the scene cannot ask; a
  scorecard counts each as a miss and says so.
