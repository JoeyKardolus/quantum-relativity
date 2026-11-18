# physics/scenes/quantum_split/definitions/

Formulas on the store's columns (whole numbers); no universe.

- One number tests the rule "a cell's share is its chance": over many splits, the
  share that settle in their largest half must equal the mean share those halves held.
  A scene that always settled in the largest half would read 1, not the mean share.
- `largest_half` breaks a tie toward the lower cell id; the store's halves come sorted
  by split and cell.
- A settling whose split was not open at the reading (it opened in the last step) has
  no share on record and is left out of `landing`, and counted.
